import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_transition(path: Path) -> np.ndarray:
    data = np.load(path)
    if {"idx", "values", "shape"}.issubset(data.files):
        dense = np.zeros(tuple(data["shape"]), dtype=np.float32)
        dense[tuple(data["idx"])] = data["values"]
        return dense
    if "data" in data.files:
        return data["data"].astype(np.float32)
    raise ValueError(f"Unsupported transition format: {path}")


def edge_set(matrix: np.ndarray) -> set:
    src, tgt = np.where(matrix > 0)
    return {(int(s), int(t)) for s, t in zip(src.tolist(), tgt.tolist())}


def row_topk_edges(matrix: np.ndarray, topk: int, threshold: float = 0.0) -> set:
    edges = set()
    for src in range(matrix.shape[0]):
        row = matrix[src]
        candidates = np.flatnonzero(row > threshold)
        candidates = candidates[candidates != src]
        if candidates.size == 0:
            continue
        if candidates.size > topk:
            scores = row[candidates]
            chosen = np.argpartition(scores, -topk)[-topk:]
            candidates = candidates[chosen]
        edges.update((src, int(tgt)) for tgt in candidates.tolist())
    return edges


def values(matrix: np.ndarray) -> np.ndarray:
    return matrix[matrix > 0].astype(np.float32)


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def maybe_load_stats(dataset_root: Path) -> dict:
    path = dataset_root / "artifacts" / "oof_candidate_stats.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_code_names(dataset_root: Path) -> dict:
    code_map_path = dataset_root / "encoded" / "code_map.pkl"
    desc_path = dataset_root / "code_description_map.json"
    idx_to_code = {}
    if code_map_path.exists():
        with code_map_path.open("rb") as f:
            code_map = pickle.load(f)
        if isinstance(code_map, dict):
            for code, idx in code_map.items():
                idx_to_code[int(idx)] = str(code)
    desc = {}
    if desc_path.exists():
        with desc_path.open("r", encoding="utf-8") as f:
            desc = json.load(f)
    return {idx: f"{code} {desc.get(code, '')}".strip() for idx, code in idx_to_code.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Final OOF candidate dataset")
    parser.add_argument("--positive-topk", type=int, default=None)
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument("--top-overlap-diff", type=int, default=100)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    dataset_root = ROOT / "data" / args.dataset
    train_dir = dataset_root / "standard" / "train"
    t_data = load_transition(train_dir / "transition_T_decay_multi.npz")
    t_sem = load_transition(train_dir / "transition_T_llm.npz")
    if t_data.shape != t_sem.shape:
        raise ValueError(f"shape mismatch: T_data={t_data.shape}, T_sem={t_sem.shape}")

    stats = maybe_load_stats(dataset_root)
    positive_topk = args.positive_topk
    if positive_topk is None:
        positive_topk = int(stats.get("positive_topk", 20))

    sem_edges = edge_set(t_sem)
    data_edges = edge_set(t_data)
    overlap = sem_edges & data_edges
    new_edges = sem_edges - data_edges
    e_pos = row_topk_edges(t_data, positive_topk, args.positive_threshold)
    sem_values = values(t_sem)
    data_values = values(t_data)
    overlap_pairs = np.asarray(list(overlap), dtype=np.int64) if overlap else np.zeros((0, 2), dtype=np.int64)
    if overlap_pairs.size:
        sem_overlap_values = t_sem[overlap_pairs[:, 0], overlap_pairs[:, 1]]
        data_overlap_values = t_data[overlap_pairs[:, 0], overlap_pairs[:, 1]]
        sem_gt_data_ratio = float(np.mean(sem_overlap_values > data_overlap_values))
    else:
        sem_overlap_values = np.asarray([], dtype=np.float32)
        data_overlap_values = np.asarray([], dtype=np.float32)
        sem_gt_data_ratio = float("nan")

    rows = t_sem.shape[0]
    nnz = len(sem_edges)
    active_rows = int(np.count_nonzero(np.sum(t_sem > 0, axis=1) > 0))
    summary = {
        "dataset": args.dataset,
        "rows": rows,
        "cols": t_sem.shape[1],
        "nnz": nnz,
        "pair_coverage": nnz / float(t_sem.size),
        "source_coverage": active_rows / float(rows),
        "active_source_count": active_rows,
        "avg_edges_per_row": nnz / float(rows),
        "overlap_edges": len(overlap),
        "new_edges": len(new_edges),
        "e_pos_count": len(e_pos),
        "e_pos_retained": len(e_pos & sem_edges),
        "e_pos_retention": len(e_pos & sem_edges) / float(len(e_pos)) if e_pos else "",
        "t_sem_mean": float(sem_values.mean()) if sem_values.size else "",
        "t_sem_median": float(np.median(sem_values)) if sem_values.size else "",
        "t_sem_p75": float(np.percentile(sem_values, 75)) if sem_values.size else "",
        "t_sem_p90": float(np.percentile(sem_values, 90)) if sem_values.size else "",
        "t_data_mean": float(data_values.mean()) if data_values.size else "",
        "t_data_median": float(np.median(data_values)) if data_values.size else "",
        "overlap_t_sem_mean": float(sem_overlap_values.mean()) if sem_overlap_values.size else "",
        "overlap_t_sem_median": float(np.median(sem_overlap_values)) if sem_overlap_values.size else "",
        "overlap_t_data_mean": float(data_overlap_values.mean()) if data_overlap_values.size else "",
        "overlap_t_data_median": float(np.median(data_overlap_values)) if data_overlap_values.size else "",
        "overlap_t_sem_gt_t_data_ratio": sem_gt_data_ratio,
        "overlap_t_sem_gt_t_data_percent": pct(sem_gt_data_ratio) if overlap_pairs.size else "",
        "positive_topk": positive_topk,
        "positive_threshold": args.positive_threshold,
    }

    out_prefix = Path(args.out_prefix) if args.out_prefix else ROOT / "analysis" / args.dataset / "final_oof_matrix"
    out_prefix = out_prefix if out_prefix.is_absolute() else ROOT / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Export largest overlap score gaps for manual inspection.
    label_names = load_code_names(dataset_root)
    detail_path = out_prefix.with_name(out_prefix.name + "_overlap_score_gap_top.csv")
    if overlap_pairs.size:
        gap = sem_overlap_values - data_overlap_values
        order = np.argsort(-np.abs(gap))[: args.top_overlap_diff]
        with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = [
                "source_idx", "source_label", "target_idx", "target_label",
                "T_data", "T_sem", "T_sem_minus_T_data", "T_sem_gt_T_data",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in order.tolist():
                src, tgt = overlap_pairs[idx].tolist()
                writer.writerow({
                    "source_idx": src,
                    "source_label": label_names.get(src, str(src)),
                    "target_idx": tgt,
                    "target_label": label_names.get(tgt, str(tgt)),
                    "T_data": float(data_overlap_values[idx]),
                    "T_sem": float(sem_overlap_values[idx]),
                    "T_sem_minus_T_data": float(gap[idx]),
                    "T_sem_gt_T_data": bool(sem_overlap_values[idx] > data_overlap_values[idx]),
                })

    print(f"summary saved to {summary_path}")
    print(f"overlap gap details saved to {detail_path}")


if __name__ == "__main__":
    main()
