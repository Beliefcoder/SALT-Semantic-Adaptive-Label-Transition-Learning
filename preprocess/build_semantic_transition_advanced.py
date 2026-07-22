import argparse
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from llama3_hidden_pipeline_utils import (
    SemanticProjector,
    batched_cosine_topk,
    build_candidate_pairs,
    compute_transition_coverage_stats,
    ensure_dir,
    infer_hidden_dataset_root,
    load_embedding_bundle,
    load_sparse_transition,
    resolve_transition_filename,
    row_topk_indices,
    save_projector_checkpoint,
    save_sparse_transition,
    save_transition_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build semantic transition prior with optional all-pair scoring and OOF overlap scoring."
    )
    parser.add_argument("--dataset", default="car_qwen25_hidden_v1")
    parser.add_argument("--embedding_path", default=None)
    parser.add_argument("--transition_variant", default="decay_multi", choices=["markov", "decay_multi", "ppmi_future"])
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--output_topk", type=int, default=20)
    parser.add_argument("--output_threshold", type=float, default=0.05)
    parser.add_argument("--score_all_pairs", action="store_true")
    parser.add_argument("--oof_overlap", action="store_true")
    parser.add_argument("--oof_folds", type=int, default=3)
    parser.add_argument("--overlap_score_scale", type=float, default=1.0,
                        help="Scale OOF scores on supervised empirical positive edges before final row-wise Top-K")
    parser.add_argument("--candidate_data_topk", type=int, default=20)
    parser.add_argument("--candidate_data_threshold", type=float, default=0.0)
    parser.add_argument("--candidate_cosine_topk", type=int, default=20)
    parser.add_argument("--candidate_random_per_row", type=int, default=4)
    parser.add_argument("--cosine_device", default="cuda")
    parser.add_argument("--cosine_chunk_size", type=int, default=256)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--artifact_name", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positive_edges(
    transition_dense: np.ndarray,
    positive_topk: int,
    positive_threshold: float,
) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for src_idx in range(transition_dense.shape[0]):
        pos = row_topk_indices(transition_dense[src_idx], positive_topk, positive_threshold, exclude=src_idx)
        edges.extend((src_idx, int(tgt)) for tgt in pos.tolist())
    return edges


def build_pairs(
    transition_dense: np.ndarray,
    positive_topk: int,
    positive_threshold: float,
    negatives_per_positive: int,
    rng: np.random.Generator,
    heldout_positive_edges: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heldout_positive_edges = heldout_positive_edges or set()
    src_list: List[int] = []
    tgt_list: List[int] = []
    cls_targets: List[float] = []
    reg_targets: List[float] = []
    num_labels = transition_dense.shape[0]
    universe = np.arange(num_labels, dtype=np.int64)

    for src_idx in range(num_labels):
        row = transition_dense[src_idx]
        pos_all = row_topk_indices(row, positive_topk, positive_threshold, exclude=src_idx)
        if pos_all.size == 0:
            continue
        pos = np.asarray(
            [int(tgt) for tgt in pos_all.tolist() if (src_idx, int(tgt)) not in heldout_positive_edges],
            dtype=np.int64,
        )
        if pos.size > 0:
            assert heldout_positive_edges.isdisjoint({(src_idx, int(tgt)) for tgt in pos.tolist()})
            src_list.extend([src_idx] * pos.size)
            tgt_list.extend(pos.tolist())
            cls_targets.extend([1.0] * pos.size)
            reg_targets.extend(row[pos].astype(np.float32).tolist())

        banned = np.zeros(num_labels, dtype=bool)
        banned[src_idx] = True
        banned[pos_all] = True
        for _, tgt in [edge for edge in heldout_positive_edges if edge[0] == src_idx]:
            banned[tgt] = True
        pool = universe[~banned]
        if pool.size == 0:
            continue
        neg_count = min(pool.size, max(1, negatives_per_positive * max(1, pos_all.size)))
        sampled = rng.choice(pool, size=neg_count, replace=False)
        sampled_edges = {(src_idx, int(tgt)) for tgt in sampled.tolist()}
        assert sampled_edges.isdisjoint(heldout_positive_edges), "held-out positive edge sampled as negative"
        src_list.extend([src_idx] * sampled.size)
        tgt_list.extend(sampled.tolist())
        cls_targets.extend([0.0] * sampled.size)
        reg_targets.extend([0.0] * sampled.size)

    if not src_list:
        raise RuntimeError("No projector training pairs were generated")
    return (
        np.asarray(src_list, dtype=np.int64),
        np.asarray(tgt_list, dtype=np.int64),
        np.asarray(cls_targets, dtype=np.float32),
        np.asarray(reg_targets, dtype=np.float32),
    )


def train_projector(
    embeddings: torch.Tensor,
    transition_dense: np.ndarray,
    args,
    device: torch.device,
    heldout_positive_edges: Optional[Set[Tuple[int, int]]] = None,
    seed_offset: int = 0,
) -> Tuple[SemanticProjector, Dict[str, float]]:
    rng = np.random.default_rng(args.seed + seed_offset)
    src_idx, tgt_idx, cls_targets, reg_targets = build_pairs(
        transition_dense=transition_dense,
        positive_topk=args.positive_topk,
        positive_threshold=args.positive_threshold,
        negatives_per_positive=args.negatives_per_positive,
        rng=rng,
        heldout_positive_edges=heldout_positive_edges,
    )
    order = rng.permutation(src_idx.shape[0])
    src_idx = src_idx[order]
    tgt_idx = tgt_idx[order]
    cls_targets = cls_targets[order]
    reg_targets = reg_targets[order]

    model = SemanticProjector(
        embedding_dim=embeddings.shape[1],
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    src_t = torch.from_numpy(src_idx)
    tgt_t = torch.from_numpy(tgt_idx)
    cls_t = torch.from_numpy(cls_targets)
    reg_t = torch.from_numpy(reg_targets)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(src_t.size(0))
        total_loss = 0.0
        total_batches = math.ceil(src_t.size(0) / args.batch_size)
        for batch_id in range(total_batches):
            sl = slice(batch_id * args.batch_size, min((batch_id + 1) * args.batch_size, src_t.size(0)))
            idx = perm[sl]
            batch_src = src_t[idx].to(device)
            batch_tgt = tgt_t[idx].to(device)
            batch_cls = cls_t[idx].to(device)
            batch_reg = reg_t[idx].to(device)
            logits = model.forward_logits(embeddings[batch_src], embeddings[batch_tgt])
            scores = torch.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, batch_cls)
            loss = loss + args.lambda_reg * F.mse_loss(scores, batch_reg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"projector epoch {epoch}/{args.epochs} - loss={total_loss / total_batches:.6f}")

    return model, {"train_pair_count": int(src_t.size(0))}


def source_stratified_folds(
    edges: Sequence[Tuple[int, int]],
    num_folds: int,
    rng: np.random.Generator,
) -> List[Set[Tuple[int, int]]]:
    by_source: Dict[int, List[int]] = {}
    for src, tgt in edges:
        by_source.setdefault(int(src), []).append(int(tgt))
    folds: List[Set[Tuple[int, int]]] = [set() for _ in range(num_folds)]
    for src, tgts in by_source.items():
        tgts_arr = np.asarray(tgts, dtype=np.int64)
        rng.shuffle(tgts_arr)
        for offset, tgt in enumerate(tgts_arr.tolist()):
            folds[offset % num_folds].add((int(src), int(tgt)))
    return folds


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    ar = rankdata(a)
    br = rankdata(b)
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def distribution_summary(values: np.ndarray, prefix: str) -> Dict[str, Optional[float]]:
    if values.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p10": None,
            f"{prefix}_p25": None,
            f"{prefix}_p75": None,
            f"{prefix}_p90": None,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def save_oof_diagnostics(path: str, full_scores: np.ndarray, oof_scores: np.ndarray) -> None:
    if full_scores.size == 0 or oof_scores.size == 0:
        return
    ensure_dir(os.path.dirname(path))
    gap = full_scores - oof_scores
    order_full = np.sort(full_scores)
    order_oof = np.sort(oof_scores)
    order_gap = np.sort(gap)
    n = full_scores.size
    ecdf_y = (np.arange(n, dtype=np.float32) + 1.0) / float(n)
    np.savez(
        path,
        full_scores=full_scores.astype(np.float32),
        oof_scores=oof_scores.astype(np.float32),
        gap=gap.astype(np.float32),
        ecdf_y=ecdf_y.astype(np.float32),
        full_ecdf_x=order_full.astype(np.float32),
        oof_ecdf_x=order_oof.astype(np.float32),
        gap_ecdf_x=order_gap.astype(np.float32),
    )


def candidate_pairs(
    embeddings_cpu: torch.Tensor,
    transition_dense: np.ndarray,
    args,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if args.score_all_pairs:
        n = transition_dense.shape[0]
        src = np.repeat(np.arange(n, dtype=np.int64), n - 1)
        tgt_rows = []
        base = np.arange(n, dtype=np.int64)
        for i in range(n):
            tgt_rows.append(base[base != i])
        tgt = np.concatenate(tgt_rows)
        return src, tgt.astype(np.int64), int(n * (n - 1))

    cosine_device = torch.device(args.cosine_device if torch.cuda.is_available() or args.cosine_device == "cpu" else "cpu")
    cosine_candidates = batched_cosine_topk(
        embeddings=embeddings_cpu,
        topk=args.candidate_cosine_topk,
        device=cosine_device,
        chunk_size=args.cosine_chunk_size,
    )
    src, tgt = build_candidate_pairs(
        transition_dense=transition_dense,
        cosine_candidates=cosine_candidates,
        data_topk=args.candidate_data_topk,
        data_threshold=args.candidate_data_threshold,
        random_candidates_per_row=args.candidate_random_per_row,
        rng=rng,
    )
    return src, tgt, int(transition_dense.shape[0] * (transition_dense.shape[0] - 1))


def score_pairs(
    projector: SemanticProjector,
    embeddings: torch.Tensor,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    score_batch_size: int,
    device: torch.device,
    label: str,
) -> np.ndarray:
    projector.eval()
    src_t = torch.from_numpy(src_idx)
    tgt_t = torch.from_numpy(tgt_idx)
    scores = np.zeros((src_idx.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, src_t.size(0), score_batch_size):
            end = min(start + score_batch_size, src_t.size(0))
            batch_src = src_t[start:end].to(device)
            batch_tgt = tgt_t[start:end].to(device)
            batch_scores = projector.forward_scores(embeddings[batch_src], embeddings[batch_tgt])
            scores[start:end] = batch_scores.detach().cpu().numpy().astype(np.float32)
            print(f"{label}: scored {end}/{src_t.size(0)}")
    return scores


def row_topk_matrix(
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    scores: np.ndarray,
    rows: int,
    cols: int,
    topk: int,
    threshold: float,
) -> np.ndarray:
    output = np.zeros((rows, cols), dtype=np.float32)
    for row_idx in range(rows):
        mask = src_idx == row_idx
        if not np.any(mask):
            continue
        row_dense = np.zeros((cols,), dtype=np.float32)
        row_dense[tgt_idx[mask]] = scores[mask]
        keep = row_topk_indices(row_dense, topk, threshold, exclude=row_idx)
        if keep.size:
            output[row_idx, keep] = row_dense[keep]
    return output


def main():
    args = parse_args()
    set_seed(args.seed)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(repo_root, "data")
    dataset_root = infer_hidden_dataset_root(data_root, args.dataset)
    artifacts_dir = os.path.join(dataset_root, "artifacts")
    ensure_dir(artifacts_dir)
    transition_path = os.path.join(dataset_root, "standard", "train", resolve_transition_filename(args.transition_variant))
    embedding_path = args.embedding_path or os.path.join(artifacts_dir, "label_embeddings_llama3_fp16.pt")
    artifact_name = args.artifact_name or (
        "transition_T_llm_oof_allpairs" if args.oof_overlap and args.score_all_pairs
        else "transition_T_llm_oof_candidate" if args.oof_overlap
        else "transition_T_llm_allpairs" if args.score_all_pairs
        else "transition_T_llm_candidate"
    )
    output_path = args.output_path or os.path.join(dataset_root, "standard", "train", "transition_T_llm.npz")
    artifact_transition = os.path.join(artifacts_dir, f"{artifact_name}.npz")
    artifact_stats = os.path.join(artifacts_dir, f"{artifact_name}_stats.json")
    artifact_oof_diag = os.path.join(artifacts_dir, f"{artifact_name}_oof_score_diagnostics.npz")
    projector_path = os.path.join(artifacts_dir, f"{artifact_name}_full_projector.pt")

    transition_dense = load_sparse_transition(transition_path)
    bundle = load_embedding_bundle(embedding_path)
    if transition_dense.shape[0] != bundle.embeddings.shape[0]:
        raise ValueError("Embedding count and transition matrix size do not match")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    embeddings_cpu = bundle.embeddings.float()
    embeddings = embeddings_cpu.to(device)
    rng = np.random.default_rng(args.seed)

    print("training full projector")
    full_projector, full_state = train_projector(embeddings, transition_dense, args, device)
    save_projector_checkpoint(projector_path, full_projector, optimizer=None, extra_state=full_state)

    src_idx, tgt_idx, candidate_universe = candidate_pairs(embeddings_cpu, transition_dense, args, rng)
    print(f"candidate/scoring pair count: {src_idx.size}")
    scores = score_pairs(full_projector, embeddings, src_idx, tgt_idx, args.score_batch_size, device, "full")

    oof_scored_edges = 0
    oof_full_scores: List[float] = []
    oof_scores: List[float] = []
    if args.oof_overlap:
        all_pos_edges = positive_edges(transition_dense, args.positive_topk, args.positive_threshold)
        pair_to_index = {(int(s), int(t)): idx for idx, (s, t) in enumerate(zip(src_idx.tolist(), tgt_idx.tolist()))}
        missing = [edge for edge in all_pos_edges if edge not in pair_to_index]
        if missing:
            preview = missing[:10]
            raise RuntimeError(
                f"E_pos is not fully covered by candidate set: missing={len(missing)} preview={preview}"
            )
        folds = source_stratified_folds(all_pos_edges, args.oof_folds, rng)
        for fold_id, heldout in enumerate(folds, start=1):
            if not heldout:
                continue
            print(f"training OOF projector fold {fold_id}/{args.oof_folds}, heldout={len(heldout)}")
            fold_projector, _ = train_projector(
                embeddings,
                transition_dense,
                args,
                device,
                heldout_positive_edges=heldout,
                seed_offset=fold_id * 1000,
            )
            fold_src = np.asarray([src for src, _ in heldout], dtype=np.int64)
            fold_tgt = np.asarray([tgt for _, tgt in heldout], dtype=np.int64)
            fold_scores = score_pairs(
                fold_projector,
                embeddings,
                fold_src,
                fold_tgt,
                args.score_batch_size,
                device,
                f"oof_fold_{fold_id}",
            )
            for src, tgt, score in zip(fold_src.tolist(), fold_tgt.tolist(), fold_scores.tolist()):
                idx = pair_to_index.get((src, tgt))
                if idx is not None:
                    oof_full_scores.append(float(scores[idx]))
                    oof_scores.append(float(score))
                    scores[idx] = float(score) * args.overlap_score_scale
                    oof_scored_edges += 1

    output_matrix = row_topk_matrix(
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        scores=scores,
        rows=transition_dense.shape[0],
        cols=transition_dense.shape[1],
        topk=args.output_topk,
        threshold=args.output_threshold,
    )
    save_sparse_transition(output_path, output_matrix)
    save_sparse_transition(artifact_transition, output_matrix)
    oof_full_arr = np.asarray(oof_full_scores, dtype=np.float32)
    oof_arr = np.asarray(oof_scores, dtype=np.float32)
    pos_retained = 0
    if args.oof_overlap and oof_scored_edges:
        pos_edge_set = set(positive_edges(transition_dense, args.positive_topk, args.positive_threshold))
        out_src, out_tgt = np.where(output_matrix > 0)
        out_edges = {(int(src), int(tgt)) for src, tgt in zip(out_src.tolist(), out_tgt.tolist())}
        pos_retained = len(pos_edge_set & out_edges)
    stats = compute_transition_coverage_stats(output_matrix)
    oof_gap_arr = oof_full_arr - oof_arr if oof_arr.size else np.asarray([], dtype=np.float32)
    save_oof_diagnostics(artifact_oof_diag, oof_full_arr, oof_arr)
    stats.update(
        {
            "dataset": args.dataset,
            "transition_variant": args.transition_variant,
            "score_all_pairs": bool(args.score_all_pairs),
            "oof_overlap": bool(args.oof_overlap),
            "oof_folds": int(args.oof_folds),
            "overlap_score_scale": float(args.overlap_score_scale),
            "positive_topk": int(args.positive_topk),
            "output_topk": int(args.output_topk),
            "output_threshold": float(args.output_threshold),
            "candidate_data_topk": int(args.candidate_data_topk),
            "candidate_cosine_topk": int(args.candidate_cosine_topk),
            "candidate_random_per_row": int(args.candidate_random_per_row),
            "oof_scored_edges": int(oof_scored_edges),
            **distribution_summary(oof_full_arr, "oof_full_score"),
            **distribution_summary(oof_arr, "oof_score"),
            **distribution_summary(oof_gap_arr, "oof_gap_full_minus_oof"),
            "oof_mean_gap_full_minus_oof": float((oof_full_arr - oof_arr).mean()) if oof_arr.size else None,
            "oof_spearman_full_vs_oof": spearman_corr(oof_full_arr, oof_arr) if oof_arr.size else None,
            "oof_positive_topk_retained": int(pos_retained),
            "oof_positive_topk_retention_ratio": float(pos_retained / oof_scored_edges) if oof_scored_edges else None,
            "scored_pair_count": int(src_idx.size),
            "candidate_coverage_ratio": float(src_idx.size / candidate_universe) if candidate_universe else 0.0,
            "output_path": output_path,
            "artifact_transition": artifact_transition,
            "artifact_oof_diagnostics": artifact_oof_diag if oof_arr.size else None,
        }
    )
    save_transition_stats(artifact_stats, stats)
    print(f"saved semantic transition matrix to {output_path}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
