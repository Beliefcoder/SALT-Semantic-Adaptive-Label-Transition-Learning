import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from scipy import sparse

from metrics import evaluate_codes
from models.model import Model
from utils import EHRDataset, format_time, load_adj


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sparse_transition(path):
    data = np.load(path)
    matrix = np.zeros(tuple(data["shape"].tolist()), dtype=data["values"].dtype)
    matrix[tuple(data["idx"])] = data["values"]
    return sparse.csr_matrix(matrix)


def matrix_summary(name, matrix):
    values = matrix.data
    nnz = int(matrix.nnz)
    total = matrix.shape[0] * matrix.shape[1]
    coverage = nnz / total if total else 0.0
    mean = float(values.mean()) if nnz else 0.0
    median = float(np.median(values)) if nnz else 0.0
    max_value = float(values.max()) if nnz else 0.0
    print(
        f"{name}: shape={matrix.shape} nnz={nnz} coverage={coverage:.6f} "
        f"mean={mean:.6f} median={median:.6f} max={max_value:.6f}"
    )


def historical_hot(data_path, code_num, use_all_history=False):
    from preprocess import load_sparse

    code_x = load_sparse(os.path.join(data_path, "code_x.npz"), return_sparse=False)
    visit_lens = np.load(os.path.join(data_path, "visit_lens.npz"))["lens"]
    result = np.zeros((code_x.shape[0], code_num), dtype=np.float32)
    for i in range(code_x.shape[0]):
        valid_len = int(visit_lens[i])
        if valid_len <= 0:
            continue
        if use_all_history:
            result[i] = (np.sum(code_x[i, :valid_len, :], axis=0) > 0).astype(np.float32)
        else:
            result[i] = code_x[i, valid_len - 1, :]
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train the fixed SALT mainline model.")
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--cache-dataset-on-gpu", action="store_true",
                        help="Optional GPU dataset cache. Disabled by default to reduce memory use.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--result-tag", default="salt_main_seed50")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    dataset_root = os.path.join("data", args.dataset)
    standard_root = os.path.join(dataset_root, "standard")
    train_path = os.path.join(standard_root, "train")
    valid_path = os.path.join(standard_root, "valid")
    test_path = os.path.join(standard_root, "test")

    code_map_path = os.path.join(dataset_root, "encoded", "code_map.pkl")
    data_transition_path = os.path.join(train_path, "transition_T_decay_multi.npz")
    sem_transition_path = os.path.join(train_path, "transition_T_llm.npz")

    print(f"dataset: {args.dataset}")
    print(f"code_map: {code_map_path}")
    t_data = load_sparse_transition(data_transition_path)
    t_sem = load_sparse_transition(sem_transition_path)
    matrix_summary("T_data", t_data)
    matrix_summary("T_sem", t_sem)
    print(f"device: {device}")
    print("mainline: LSTM + dual-prior propagation + sample-adaptive fusion + TCRF + RIF")
    print(f"batch_size: {args.batch_size}, epochs: {args.epochs}, seed: {args.seed}")
    print(f"gpu_dataset_cache: {bool(args.cache_dataset_on_gpu and use_cuda)}")

    code_adj = load_adj(standard_root, device=device)
    code_num = int(code_adj.shape[0])

    cache_on_device = bool(args.cache_dataset_on_gpu and use_cuda)
    train_data = EHRDataset(train_path, label="m", batch_size=args.batch_size, shuffle=True,
                            device=device, cache_on_device=cache_on_device)
    valid_data = EHRDataset(valid_path, label="m", batch_size=args.batch_size, shuffle=False,
                            device=device, cache_on_device=cache_on_device)
    test_data = EHRDataset(test_path, label="m", batch_size=args.batch_size, shuffle=False,
                           device=device, cache_on_device=cache_on_device)

    valid_historical = historical_hot(valid_path, code_num, use_all_history=False)
    test_historical = historical_hot(test_path, code_num, use_all_history=False)
    valid_all_historical = historical_hot(valid_path, code_num, use_all_history=True)
    test_all_historical = historical_hot(test_path, code_num, use_all_history=True)
    train_label_freq = np.asarray(train_data.label().sum(axis=0)).reshape(-1)

    model = Model(
        code_num=code_num,
        code_size=48,
        adj=code_adj,
        graph_size=32,
        hidden_size=150,
        t_attention_size=32,
        t_output_size=150,
        output_size=code_num,
        dropout_rate=0.47,
        transition_T=t_data,
        transition_T_llm=t_sem,
        use_transition=False,
        use_llm_transition=True,
        llm_alpha=0.5,
        fusion_type="fusion",
        max_seq_len=70,
        use_tcn=False,
        tcn_num_layers=2,
        tcn_kernel_size=3,
        stream_fusion_type="tcrf",
        use_multi_scale_agg=False,
        tcrf_type="improved",
        use_packed_sequence=True,
        use_masked_attention=True,
        transition_mode="feature_fusion",
        prior_lambda=0.5,
        transition_fusion_variant="sample_adaptive",
        use_stream_feature_norm=True,
        use_dual_stream_alignment=False,
        use_residual_interaction_fusion=True,
        use_label_wise_gate=False,
        use_label_graph_head=False,
        use_gated_residual_interaction_fusion=False,
        use_cross_attention_fusion=False,
        simple_fusion_mode="sum_mul",
        context_branch="retain_attention",
        transition_gate_mode="sample_adaptive",
        shared_transition_projection=False,
    ).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"optimizer: AdamW lr=5e-5 weight_decay=0.01")
    print(f"trainable_parameters: {total_params}")

    result_dir = os.path.join("results", args.dataset)
    os.makedirs(result_dir, exist_ok=True)
    valid_csv = os.path.join(result_dir, f"{args.result_tag}_validation.csv")
    test_csv = os.path.join(result_dir, f"{args.result_tag}_test.csv")
    checkpoint_dir = os.path.join("checkpoints", args.dataset, args.result_tag)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
    if args.save_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    output_size = code_num

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1} / {args.epochs}:")
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        start = time.time()

        for step in range(steps):
            code_x, visit_lens, divided, y, neighbors = train_data[step]
            del divided, neighbors
            optimizer.zero_grad(set_to_none=True)
            logits = model(code_x, visit_lens).squeeze()
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)

        train_data.on_epoch_end()
        print(
            f"    Step {steps} / {steps}, time cost: {format_time(time.time() - start)}, "
            f"loss: {total_loss / max(total_num, 1):.4f}"
        )

        _, valid_f1 = evaluate_codes(
            code_adj,
            model,
            valid_data,
            loss_fn,
            output_size,
            historical=valid_historical,
            epoch=epoch,
            save_csv=True,
            csv_path=valid_csv,
            train_label_freq=train_label_freq,
            split_name="Validation",
            all_historical=valid_all_historical,
        )
        if epoch == 0 or valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            if args.save_checkpoint:
                torch.save(
                    {"epoch": best_epoch, "model_state_dict": best_state, "best_f1": best_f1},
                    best_checkpoint_path,
                )

    print(f"Best epoch: {best_epoch}, valid score: {best_f1:.4f}")
    print("Evaluating best model on test data ...")
    model.load_state_dict(best_state)
    test_loss, test_f1 = evaluate_codes(
        code_adj,
        model,
        test_data,
        loss_fn,
        output_size,
        historical=test_historical,
        epoch=best_epoch - 1,
        save_csv=True,
        csv_path=test_csv,
        train_label_freq=train_label_freq,
        split_name="Test",
        all_historical=test_all_historical,
    )
    print(f"Final test evaluation from epoch {best_epoch}: test_loss={test_loss:.4f}, test_f1={test_f1:.4f}")
    print(f"test results: {test_csv}")


if __name__ == "__main__":
    main()
