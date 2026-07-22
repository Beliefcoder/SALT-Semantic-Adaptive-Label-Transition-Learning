import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from llama3_hidden_pipeline_utils import (
    SemanticProjector,
    ensure_dir,
    infer_hidden_dataset_root,
    load_embedding_bundle,
    load_or_build_full_transition_dense,
    load_sparse_transition,
    resolve_transition_filename,
    row_topk_indices,
    save_projector_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train semantic projector from label embeddings and T_data.")
    parser.add_argument("--source_dataset", type=str, default="car")
    parser.add_argument("--variant_dataset", type=str, default="car_llama3_hidden_v1")
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--transition_variant", type=str, default="decay_multi", choices=["markov", "decay_multi", "ppmi_future"])
    parser.add_argument("--positive_topk", type=int, default=20)
    parser.add_argument("--positive_threshold", type=float, default=0.0)
    parser.add_argument("--negatives_per_positive", type=int, default=2)
    parser.add_argument("--bottleneck_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--loss_mode", type=str, default="strict", choices=["strict", "weak", "masked_bpr"],
                        help="strict: BCE + MSE on all sampled pairs. weak: strong positives, weak negatives, MSE only on positives.")
    parser.add_argument("--mask_ratio", type=float, default=0.0,
                        help="Per-source empirical-edge mask ratio for masked reconstruction. 0 disables masking.")
    parser.add_argument("--min_edges_for_mask", type=int, default=2,
                        help="Minimum positive out-degree required to mask at least one empirical edge.")
    parser.add_argument("--positive_edge_source", type=str, default="all",
                        choices=["all", "masked", "masked_plus_anchor"],
                        help="Which empirical positive edges are used for projector training when --mask_ratio > 0.")
    parser.add_argument("--anchor_positive_weight", type=float, default=0.2,
                        help="Positive sample/ranking weight for anchor positives under masked_plus_anchor.")
    parser.add_argument("--lambda_neg", type=float, default=0.05,
                        help="Negative BCE weight used by --loss_mode weak.")
    parser.add_argument("--lambda_bce", type=float, default=0.1,
                        help="Auxiliary BCE weight for --loss_mode masked_bpr.")
    parser.add_argument("--lambda_rank", type=float, default=0.0,
                        help="Optional pairwise ranking loss weight used by --loss_mode weak.")
    parser.add_argument("--rank_loss_type", type=str, default="hinge", choices=["hinge", "bpr"],
                        help="Ranking loss type for weak/masked_bpr modes.")
    parser.add_argument("--rank_margin", type=float, default=0.2)
    parser.add_argument("--target_debias", action="store_true",
                        help="Use inverse target-frequency weights for negative sampling and BCE.")
    parser.add_argument("--target_debias_power", type=float, default=0.5,
                        help="Debias weight power. 0.5 means 1/sqrt(freq+1).")
    parser.add_argument("--negative_sampling", type=str, default="uniform",
                        choices=["uniform", "target_debiased"],
                        help="Negative sampling distribution.")
    parser.add_argument("--source_group_batches", action="store_true",
                        help="Keep same-source pairs close in a batch so source-wise ranking loss is effective.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--use_full_transition_for_llm", action="store_true")
    parser.add_argument("--full_transition_path", type=str, default=None)
    parser.add_argument("--full_future_steps", type=int, default=3)
    parser.add_argument("--full_decay", type=float, default=0.7)
    parser.add_argument("--full_ppmi_positive_threshold", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_pairs(
    transition_dense: np.ndarray,
    positive_topk: int,
    positive_threshold: float,
    negatives_per_positive: int,
    rng: np.random.Generator,
    negative_sampling: str = "uniform",
    target_sampling_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_list: List[int] = []
    tgt_list: List[int] = []
    cls_targets: List[float] = []
    reg_targets: List[float] = []
    num_labels = transition_dense.shape[0]
    universe = np.arange(num_labels, dtype=np.int64)

    for src_idx in range(num_labels):
        row = transition_dense[src_idx]
        pos = row_topk_indices(row, positive_topk, positive_threshold, exclude=src_idx)
        if pos.size == 0:
            continue

        src_list.extend([src_idx] * pos.size)
        tgt_list.extend(pos.tolist())
        cls_targets.extend([1.0] * pos.size)
        reg_targets.extend(row[pos].astype(np.float32).tolist())

        neg_count = min((num_labels - 1) - pos.size, max(1, negatives_per_positive * pos.size))
        if neg_count <= 0:
            continue
        banned = np.zeros(num_labels, dtype=bool)
        banned[src_idx] = True
        banned[pos] = True
        pool = universe[~banned]
        if pool.size == 0:
            continue
        sample_size = min(neg_count, pool.size)
        if negative_sampling == "target_debiased" and target_sampling_weights is not None:
            probs = target_sampling_weights[pool].astype(np.float64)
            probs = probs / probs.sum() if probs.sum() > 0 else None
            sampled = rng.choice(pool, size=sample_size, replace=False, p=probs)
        else:
            sampled = rng.choice(pool, size=sample_size, replace=False)
        src_list.extend([src_idx] * sampled.size)
        tgt_list.extend(sampled.tolist())
        cls_targets.extend([0.0] * sampled.size)
        reg_targets.extend([0.0] * sampled.size)

    return (
        np.asarray(src_list, dtype=np.int64),
        np.asarray(tgt_list, dtype=np.int64),
        np.asarray(cls_targets, dtype=np.float32),
        np.asarray(reg_targets, dtype=np.float32),
    )


def split_empirical_edges_by_source(
    transition_dense: np.ndarray,
    positive_topk: int,
    positive_threshold: float,
    mask_ratio: float,
    min_edges_for_mask: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    anchor_by_source: List[np.ndarray] = []
    masked_by_source: List[np.ndarray] = []
    for src_idx in range(transition_dense.shape[0]):
        pos = row_topk_indices(
            transition_dense[src_idx],
            positive_topk,
            positive_threshold,
            exclude=src_idx,
        )
        if pos.size < min_edges_for_mask or mask_ratio <= 0:
            anchor_by_source.append(pos.astype(np.int64))
            masked_by_source.append(np.asarray([], dtype=np.int64))
            continue
        num_mask = max(1, int(round(pos.size * mask_ratio)))
        num_mask = min(num_mask, pos.size - 1)
        masked = rng.choice(pos, size=num_mask, replace=False).astype(np.int64)
        masked_set = set(masked.tolist())
        anchor = np.asarray([dst for dst in pos.tolist() if dst not in masked_set], dtype=np.int64)
        anchor_by_source.append(anchor)
        masked_by_source.append(masked)
    return anchor_by_source, masked_by_source


def build_masked_reconstruction_pairs(
    transition_dense: np.ndarray,
    positive_topk: int,
    positive_threshold: float,
    negatives_per_positive: int,
    rng: np.random.Generator,
    mask_ratio: float,
    min_edges_for_mask: int,
    positive_edge_source: str,
    anchor_positive_weight: float,
    negative_sampling: str = "uniform",
    target_sampling_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_by_source, masked_by_source = split_empirical_edges_by_source(
        transition_dense=transition_dense,
        positive_topk=positive_topk,
        positive_threshold=positive_threshold,
        mask_ratio=mask_ratio,
        min_edges_for_mask=min_edges_for_mask,
        rng=rng,
    )
    num_labels = transition_dense.shape[0]
    universe = np.arange(num_labels, dtype=np.int64)
    out_degrees = np.asarray(
        [anchor_by_source[i].size + masked_by_source[i].size for i in range(num_labels)],
        dtype=np.float32,
    )
    source_weights = 1.0 / np.sqrt(out_degrees + 1.0)
    source_weights = source_weights / max(float(source_weights.mean()), 1e-12)

    src_list: List[int] = []
    tgt_list: List[int] = []
    cls_targets: List[float] = []
    reg_targets: List[float] = []
    src_weights: List[float] = []
    pair_weights: List[float] = []

    for src_idx in range(num_labels):
        anchor = anchor_by_source[src_idx]
        masked = masked_by_source[src_idx]
        if positive_edge_source == "masked":
            pos = masked
            pos_weights = np.ones((pos.size,), dtype=np.float32)
        elif positive_edge_source == "masked_plus_anchor":
            if masked.size:
                pos = np.concatenate([masked, anchor])
                pos_weights = np.concatenate([
                    np.ones((masked.size,), dtype=np.float32),
                    np.full((anchor.size,), float(anchor_positive_weight), dtype=np.float32),
                ])
            else:
                pos = anchor
                pos_weights = np.full((anchor.size,), float(anchor_positive_weight), dtype=np.float32)
        else:
            pos = np.concatenate([masked, anchor]) if masked.size else anchor
            pos_weights = np.ones((pos.size,), dtype=np.float32)
        if pos.size == 0:
            continue

        src_list.extend([src_idx] * pos.size)
        tgt_list.extend(pos.tolist())
        cls_targets.extend([1.0] * pos.size)
        reg_targets.extend(transition_dense[src_idx, pos].astype(np.float32).tolist())
        src_weights.extend([float(source_weights[src_idx])] * pos.size)
        pair_weights.extend(pos_weights.astype(np.float32).tolist())

        full_observed = np.concatenate([anchor, masked]) if masked.size else anchor
        banned = np.zeros(num_labels, dtype=bool)
        banned[src_idx] = True
        banned[full_observed] = True
        pool = universe[~banned]
        if pool.size == 0:
            continue
        neg_count = min(pool.size, max(1, negatives_per_positive * pos.size))
        if negative_sampling == "target_debiased" and target_sampling_weights is not None:
            probs = target_sampling_weights[pool].astype(np.float64)
            probs = probs / probs.sum() if probs.sum() > 0 else None
            sampled = rng.choice(pool, size=neg_count, replace=False, p=probs)
        else:
            sampled = rng.choice(pool, size=neg_count, replace=False)
        src_list.extend([src_idx] * sampled.size)
        tgt_list.extend(sampled.tolist())
        cls_targets.extend([0.0] * sampled.size)
        reg_targets.extend([0.0] * sampled.size)
        src_weights.extend([float(source_weights[src_idx])] * sampled.size)
        pair_weights.extend([1.0] * sampled.size)

    return (
        np.asarray(src_list, dtype=np.int64),
        np.asarray(tgt_list, dtype=np.int64),
        np.asarray(cls_targets, dtype=np.float32),
        np.asarray(reg_targets, dtype=np.float32),
        np.asarray(src_weights, dtype=np.float32),
        np.asarray(pair_weights, dtype=np.float32),
    )


def load_label_frequency(dataset_root: str) -> Optional[np.ndarray]:
    label_path = os.path.join(dataset_root, "standard", "train", "code_y.npz")
    if not os.path.exists(label_path):
        return None
    labels = load_sparse_transition(label_path)
    return labels.sum(axis=0).astype(np.float32)


def build_target_debias_weights(freq: Optional[np.ndarray], num_labels: int, power: float) -> np.ndarray:
    if freq is None:
        freq = np.zeros((num_labels,), dtype=np.float32)
    weights = 1.0 / np.power(freq.astype(np.float32) + 1.0, power)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def source_wise_rank_loss(
    scores: torch.Tensor,
    logits: torch.Tensor,
    cls_targets: torch.Tensor,
    src_indices: torch.Tensor,
    margin: float,
    rank_loss_type: str = "hinge",
    source_weights: Optional[torch.Tensor] = None,
    pair_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    losses = []
    unique_sources = torch.unique(src_indices)
    for source in unique_sources:
        mask = src_indices == source
        source_scores = scores[mask]
        source_logits = logits[mask]
        source_cls = cls_targets[mask]
        source_weight = None if source_weights is None else source_weights[mask][0]
        source_pair_weights = None if pair_weights is None else pair_weights[mask]
        if rank_loss_type == "bpr":
            pos_values = source_logits[source_cls > 0.5]
            neg_values = source_logits[source_cls <= 0.5]
        else:
            pos_values = source_scores[source_cls > 0.5]
            neg_values = source_scores[source_cls <= 0.5]
        if pos_values.numel() == 0 or neg_values.numel() == 0:
            continue
        if rank_loss_type == "bpr":
            loss_matrix = -F.logsigmoid(pos_values.unsqueeze(1) - neg_values.unsqueeze(0))
            if source_pair_weights is not None:
                pos_weights = source_pair_weights[source_cls > 0.5].clamp_min(0.0)
                loss_matrix = loss_matrix * pos_weights.unsqueeze(1)
                source_loss = loss_matrix.sum() / pos_weights.sum().clamp_min(1e-12) / neg_values.numel()
            else:
                source_loss = loss_matrix.mean()
        else:
            loss_matrix = F.relu(margin - pos_values.unsqueeze(1) + neg_values.unsqueeze(0))
            if source_pair_weights is not None:
                pos_weights = source_pair_weights[source_cls > 0.5].clamp_min(0.0)
                loss_matrix = loss_matrix * pos_weights.unsqueeze(1)
                source_loss = loss_matrix.sum() / pos_weights.sum().clamp_min(1e-12) / neg_values.numel()
            else:
                source_loss = loss_matrix.mean()
        if source_weight is not None:
            source_loss = source_loss * source_weight
        losses.append(source_loss)
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def main():
    args = parse_args()
    set_seed(args.seed)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, "data")
    source_dataset_root = os.path.join(data_root, args.source_dataset)
    target_dataset_root = infer_hidden_dataset_root(data_root, args.variant_dataset)
    artifacts_dir = os.path.join(target_dataset_root, "artifacts")
    ensure_dir(artifacts_dir)

    embedding_path = args.embedding_path or os.path.join(artifacts_dir, "label_embeddings_llama3_fp16.pt")
    output_path = args.output_path or os.path.join(artifacts_dir, "semantic_projector.pt")
    bundle = load_embedding_bundle(embedding_path)
    if args.use_full_transition_for_llm:
        transition_dense, full_transition_path, full_transition_stats = load_or_build_full_transition_dense(
            dataset_root=source_dataset_root,
            artifacts_dir=artifacts_dir,
            transition_variant=args.transition_variant,
            future_steps=args.full_future_steps,
            decay=args.full_decay,
            ppmi_positive_threshold=args.full_ppmi_positive_threshold,
            artifact_path=args.full_transition_path,
        )
        print(
            "using full transition for projector: "
            f"{full_transition_path} | nonzero={full_transition_stats['nonzero_pairs']} "
            f"| coverage={full_transition_stats['coverage_ratio']:.6f}"
        )
    else:
        transition_path = os.path.join(
            source_dataset_root,
            "standard",
            "train",
            resolve_transition_filename(args.transition_variant),
        )
        transition_dense = load_sparse_transition(transition_path)
    if transition_dense.shape[0] != bundle.embeddings.shape[0]:
        raise ValueError("Embedding count and transition matrix size do not match")

    rng = np.random.default_rng(args.seed)
    target_freq = load_label_frequency(source_dataset_root)
    target_weights = build_target_debias_weights(
        target_freq,
        transition_dense.shape[0],
        args.target_debias_power,
    )
    if args.mask_ratio > 0 or args.loss_mode == "masked_bpr":
        src_idx, tgt_idx, cls_targets, reg_targets, source_weights, pair_weights = build_masked_reconstruction_pairs(
            transition_dense=transition_dense,
            positive_topk=args.positive_topk,
            positive_threshold=args.positive_threshold,
            negatives_per_positive=args.negatives_per_positive,
            rng=rng,
            mask_ratio=args.mask_ratio if args.mask_ratio > 0 else 0.2,
            min_edges_for_mask=args.min_edges_for_mask,
            positive_edge_source=args.positive_edge_source,
            anchor_positive_weight=args.anchor_positive_weight,
            negative_sampling=args.negative_sampling,
            target_sampling_weights=target_weights if args.target_debias else None,
        )
    else:
        src_idx, tgt_idx, cls_targets, reg_targets = build_training_pairs(
            transition_dense=transition_dense,
            positive_topk=args.positive_topk,
            positive_threshold=args.positive_threshold,
            negatives_per_positive=args.negatives_per_positive,
            rng=rng,
            negative_sampling=args.negative_sampling,
            target_sampling_weights=target_weights if args.target_debias else None,
        )
        source_weights = np.ones_like(cls_targets, dtype=np.float32)
        pair_weights = np.ones_like(cls_targets, dtype=np.float32)
    if src_idx.size == 0:
        raise RuntimeError("No training pairs were generated")

    if args.source_group_batches:
        order = np.lexsort((1.0 - cls_targets, src_idx))
    else:
        order = rng.permutation(src_idx.shape[0])
    src_idx = src_idx[order]
    tgt_idx = tgt_idx[order]
    cls_targets = cls_targets[order]
    reg_targets = reg_targets[order]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    embeddings = bundle.embeddings.float().to(device)
    model = SemanticProjector(
        embedding_dim=embeddings.shape[1],
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    src_idx_t = torch.from_numpy(src_idx)
    tgt_idx_t = torch.from_numpy(tgt_idx)
    cls_targets_t = torch.from_numpy(cls_targets)
    reg_targets_t = torch.from_numpy(reg_targets)
    source_weights_t = torch.from_numpy(source_weights)
    pair_weights_t = torch.from_numpy(pair_weights)
    target_weights_t = torch.from_numpy(target_weights).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.source_group_batches:
            perm = torch.arange(src_idx_t.size(0))
        else:
            perm = torch.randperm(src_idx_t.size(0))
        total_loss = 0.0
        total_batches = math.ceil(src_idx_t.size(0) / args.batch_size)
        for batch_id in range(total_batches):
            sl = slice(batch_id * args.batch_size, min((batch_id + 1) * args.batch_size, src_idx_t.size(0)))
            batch_index = perm[sl]
            batch_src = src_idx_t[batch_index].to(device)
            batch_tgt = tgt_idx_t[batch_index].to(device)
            batch_cls = cls_targets_t[batch_index].to(device)
            batch_reg = reg_targets_t[batch_index].to(device)
            batch_source_weight = source_weights_t[batch_index].to(device)
            batch_pair_weight = pair_weights_t[batch_index].to(device)
            batch_weight = target_weights_t[batch_tgt].to(device) if args.target_debias else None

            lhs = embeddings[batch_src]
            rhs = embeddings[batch_tgt]
            logits = model.forward_logits(lhs, rhs)
            scores = torch.sigmoid(logits)
            if args.loss_mode == "strict":
                loss_cls = F.binary_cross_entropy_with_logits(logits, batch_cls)
                loss_reg = F.mse_loss(scores, batch_reg)
                loss = loss_cls + args.lambda_reg * loss_reg
            elif args.loss_mode == "weak":
                pos_mask = batch_cls > 0.5
                neg_mask = ~pos_mask
                loss_terms = []
                if pos_mask.any():
                    pos_logits = logits[pos_mask]
                    pos_scores = scores[pos_mask]
                    pos_targets = torch.ones_like(pos_logits)
                    pos_weight = batch_weight[pos_mask] if batch_weight is not None else None
                    loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_targets, weight=pos_weight)
                    loss_pos_reg = F.mse_loss(pos_scores, batch_reg[pos_mask])
                    loss_terms.append(loss_pos + args.lambda_reg * loss_pos_reg)
                if neg_mask.any() and args.lambda_neg > 0:
                    neg_logits = logits[neg_mask]
                    neg_targets = torch.zeros_like(neg_logits)
                    neg_weight = batch_weight[neg_mask] if batch_weight is not None else None
                    loss_neg = F.binary_cross_entropy_with_logits(neg_logits, neg_targets, weight=neg_weight)
                    loss_terms.append(args.lambda_neg * loss_neg)
                if pos_mask.any() and neg_mask.any() and args.lambda_rank > 0:
                    rank_loss = source_wise_rank_loss(
                        scores,
                        logits,
                        batch_cls,
                        batch_src,
                        args.rank_margin,
                        rank_loss_type=args.rank_loss_type,
                        pair_weights=batch_pair_weight,
                    )
                    loss_terms.append(args.lambda_rank * rank_loss)
                if not loss_terms:
                    loss = F.binary_cross_entropy_with_logits(logits, batch_cls)
                else:
                    loss = sum(loss_terms)
            else:
                rank_loss = source_wise_rank_loss(
                    scores,
                    logits,
                    batch_cls,
                    batch_src,
                    args.rank_margin,
                    rank_loss_type="bpr",
                    source_weights=batch_source_weight,
                    pair_weights=batch_pair_weight,
                )
                pos_mask = batch_cls > 0.5
                neg_mask = ~pos_mask
                bce_terms = []
                if pos_mask.any():
                    bce_terms.append(F.binary_cross_entropy_with_logits(
                        logits[pos_mask],
                        torch.ones_like(logits[pos_mask]),
                        weight=batch_pair_weight[pos_mask],
                    ))
                if neg_mask.any() and args.lambda_neg > 0:
                    bce_terms.append(args.lambda_neg * F.binary_cross_entropy_with_logits(
                        logits[neg_mask],
                        torch.zeros_like(logits[neg_mask]),
                    ))
                bce_loss = sum(bce_terms) if bce_terms else logits.new_tensor(0.0)
                loss = rank_loss + args.lambda_bce * bce_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / total_batches
        print(f"epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        sample_count = min(2048, src_idx_t.size(0))
        sample_src = src_idx_t[:sample_count].to(device)
        sample_tgt = tgt_idx_t[:sample_count].to(device)
        sample_scores = model.forward_scores(embeddings[sample_src], embeddings[sample_tgt]).mean().item()

    save_projector_checkpoint(
        output_path,
        model,
        optimizer=None,
        extra_state={
            "source_dataset": args.source_dataset,
            "variant_dataset": args.variant_dataset,
            "transition_variant": args.transition_variant,
            "use_full_transition_for_llm": bool(args.use_full_transition_for_llm),
            "positive_topk": args.positive_topk,
            "negatives_per_positive": args.negatives_per_positive,
            "bottleneck_dim": args.bottleneck_dim,
            "loss_mode": args.loss_mode,
            "mask_ratio": args.mask_ratio,
            "min_edges_for_mask": args.min_edges_for_mask,
            "positive_edge_source": args.positive_edge_source,
            "anchor_positive_weight": args.anchor_positive_weight,
            "lambda_neg": args.lambda_neg,
            "lambda_bce": args.lambda_bce,
            "lambda_reg": args.lambda_reg,
            "lambda_rank": args.lambda_rank,
            "rank_margin": args.rank_margin,
            "rank_loss_type": args.rank_loss_type,
            "target_debias": bool(args.target_debias),
            "target_debias_power": args.target_debias_power,
            "negative_sampling": args.negative_sampling,
            "source_group_batches": bool(args.source_group_batches),
            "train_pair_count": int(src_idx_t.size(0)),
            "sample_score_mean": sample_scores,
        },
    )
    print(f"saved projector to {output_path}")


if __name__ == "__main__":
    main()
