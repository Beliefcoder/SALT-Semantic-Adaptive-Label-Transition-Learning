import os
import sys
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import load_sparse


def _normalize_and_filter(matrix, row_base, threshold_value=0.01, topk=15):
    row_base = np.asarray(row_base, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    prob = np.zeros_like(matrix, dtype=np.float32)
    valid_rows = row_base > 0
    if np.any(valid_rows):
        prob[valid_rows] = matrix[valid_rows] / row_base[valid_rows, None]

    filtered = np.zeros_like(prob, dtype=np.float32)
    for i in range(prob.shape[0]):
        row = prob[i]
        if topk is None or topk >= row.shape[0]:
            candidate_indices = np.where(row > threshold_value)[0]
        else:
            candidate_indices = np.argsort(row)[-topk:]
        for j in candidate_indices:
            if row[j] > threshold_value:
                filtered[i, j] = row[j]
    return sparse.csr_matrix(filtered)


def generate_transition_matrix(data_path, code_num, threshold_value=0.01, topk=5):
    """
    生成时序转移矩阵 T[i,j] = P(故障j | 故障i)

    Args:
        data_path: 数据路径
        code_num: 故障代码总数
        threshold_value: 阈值，低于此值的转移概率被过滤
        topk: 每个故障最多保留topk个最高转移概率

    Returns:
        T: 转移矩阵 (code_num, code_num)
        T_counts: 转移计数矩阵
        code_freq: 每个故障的出现频率
    """
    code_x = load_sparse(os.path.join(data_path, 'code_x.npz'))
    visit_lens = np.load(os.path.join(data_path, 'visit_lens.npz'))['lens']

    T_counts = np.zeros((code_num, code_num), dtype=np.float32)
    code_freq = np.zeros(code_num, dtype=np.float32)

    for i in range(len(code_x)):
        visit_seq = code_x[i]
        visit_len = int(visit_lens[i])

        for t in range(visit_len - 1):
            current_codes = visit_seq[t]
            next_codes = visit_seq[t + 1]

            for code_i in np.where(current_codes > 0)[0]:
                code_freq[code_i] += 1
                for code_j in np.where(next_codes > 0)[0]:
                    T_counts[code_i, code_j] += 1

    T_prob = np.zeros((code_num, code_num), dtype=np.float32)
    for i in range(code_num):
        if code_freq[i] > 0:
            T_prob[i] = T_counts[i] / code_freq[i]

    T_filtered = np.zeros((code_num, code_num), dtype=np.float32)
    for i in range(code_num):
        row = T_prob[i]
        topk_indices = np.argsort(row)[-topk:]
        for j in topk_indices:
            if row[j] > threshold_value:
                T_filtered[i, j] = row[j]

    T_sparse = sparse.csr_matrix(T_filtered)

    return T_sparse, T_counts, code_freq


def generate_decay_multi_transition_matrix(data_path, code_num, threshold_value=0.01, topk=15,
                                           future_steps=3, decay=0.7):
    code_x = load_sparse(os.path.join(data_path, 'code_x.npz'), return_sparse=False)
    visit_lens = np.load(os.path.join(data_path, 'visit_lens.npz'))['lens']

    weighted_counts = np.zeros((code_num, code_num), dtype=np.float32)
    code_freq = np.zeros(code_num, dtype=np.float32)

    for i in range(len(code_x)):
        visit_seq = code_x[i]
        visit_len = int(visit_lens[i])

        for t in range(visit_len - 1):
            current_indices = np.where(visit_seq[t] > 0)[0]
            if len(current_indices) == 0:
                continue

            code_freq[current_indices] += 1.0
            max_future = min(visit_len, t + future_steps + 1)
            for future_t in range(t + 1, max_future):
                future_indices = np.where(visit_seq[future_t] > 0)[0]
                if len(future_indices) == 0:
                    continue
                weight = decay ** (future_t - t - 1)
                weighted_counts[np.ix_(current_indices, future_indices)] += weight

    T_sparse = _normalize_and_filter(weighted_counts, code_freq, threshold_value=threshold_value, topk=topk)
    return T_sparse, weighted_counts, code_freq


def generate_ppmi_future_transition_matrix(data_path, code_num, threshold_value=0.01, topk=15,
                                           future_steps=3, positive_threshold=0.0):
    code_x = load_sparse(os.path.join(data_path, 'code_x.npz'), return_sparse=False)
    visit_lens = np.load(os.path.join(data_path, 'visit_lens.npz'))['lens']

    pair_counts = np.zeros((code_num, code_num), dtype=np.float64)
    source_counts = np.zeros(code_num, dtype=np.float64)
    target_counts = np.zeros(code_num, dtype=np.float64)

    for i in range(len(code_x)):
        visit_seq = code_x[i]
        visit_len = int(visit_lens[i])

        for t in range(visit_len - 1):
            current_indices = np.where(visit_seq[t] > 0)[0]
            if len(current_indices) == 0:
                continue

            future_union = np.zeros(code_num, dtype=bool)
            max_future = min(visit_len, t + future_steps + 1)
            for future_t in range(t + 1, max_future):
                future_union |= visit_seq[future_t] > 0

            future_indices = np.where(future_union)[0]
            if len(future_indices) == 0:
                continue

            source_counts[current_indices] += 1.0
            target_counts[future_indices] += 1.0
            pair_counts[np.ix_(current_indices, future_indices)] += 1.0

    total_pairs = pair_counts.sum()
    if total_pairs <= 0:
        return sparse.csr_matrix((code_num, code_num)), pair_counts, source_counts

    source_prob = source_counts / max(source_counts.sum(), 1.0)
    target_prob = target_counts / max(target_counts.sum(), 1.0)
    joint_prob = pair_counts / total_pairs

    denom = np.outer(source_prob, target_prob)
    with np.errstate(divide='ignore', invalid='ignore'):
        ppmi = np.log((joint_prob + 1e-12) / (denom + 1e-12))
    ppmi = np.maximum(ppmi, positive_threshold).astype(np.float32)

    row_sums = ppmi.sum(axis=1).astype(np.float32)
    T_sparse = _normalize_and_filter(ppmi, row_sums, threshold_value=threshold_value, topk=topk)
    return T_sparse, pair_counts, row_sums


def save_sparse_transition(path, T):
    T_array = T.toarray() if sparse.issparse(T) else T
    idx = np.where(T_array > 0)
    values = T_array[idx]
    np.savez(path, idx=idx, values=values, shape=T_array.shape)


def fuse_matrices(T_data_path, T_llm_path, output_path, alpha=0.5, threshold=0.01, topk=15):
    T_data = np.load(T_data_path)
    T_data_arr = np.zeros(T_data['shape'], dtype=np.float32)
    T_data_arr[tuple(T_data['idx'])] = T_data['values']
    
    if T_llm_path and os.path.exists(T_llm_path):
        T_llm = np.load(T_llm_path)
        T_llm_arr = np.zeros(T_llm['shape'], dtype=np.float32)
        T_llm_arr[tuple(T_llm['idx'])] = T_llm['values']
    else:
        T_llm_arr = None
    
    T_fused = np.zeros_like(T_data_arr)
    for i in range(T_fused.shape[0]):
        if T_llm_arr is not None:
            row_fused = alpha * T_data_arr[i] + (1 - alpha) * T_llm_arr[i]
        else:
            row_fused = T_data_arr[i]
        
        topk_indices = np.argsort(row_fused)[-topk:]
        for j in topk_indices:
            if row_fused[j] > threshold:
                T_fused[i, j] = row_fused[j]
    
    save_sparse_transition(output_path, T_fused)
    return T_fused


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='car')
    parser.add_argument('--variant', type=str, default='markov',
                        choices=['markov', 'decay_multi', 'ppmi_future', 'all'])
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--future_steps', type=int, default=3)
    parser.add_argument('--decay', type=float, default=0.7)
    parser.add_argument('--ppmi_positive_threshold', type=float, default=0.0)
    args = parser.parse_args()

    dataset = args.dataset
    dataset_path = os.path.join('data', dataset, 'standard')

    code_x = load_sparse(os.path.join(dataset_path, 'train', 'code_x.npz'))
    code_num = code_x.shape[-1]
    print(f"Detected code_num from data: {code_num}")

    splits = ['train', 'valid', 'test']
    variants = ['markov', 'decay_multi', 'ppmi_future'] if args.variant == 'all' else [args.variant]

    for variant in variants:
        print(f"Generating transition matrix variant: {variant}")
        if variant == 'markov':
            T_sparse, _, row_base = generate_transition_matrix(
                os.path.join(dataset_path, 'train'),
                code_num,
                args.threshold,
                args.topk
            )
            output_name = 'transition_T.npz'
        elif variant == 'decay_multi':
            T_sparse, _, row_base = generate_decay_multi_transition_matrix(
                os.path.join(dataset_path, 'train'),
                code_num,
                threshold_value=args.threshold,
                topk=args.topk,
                future_steps=args.future_steps,
                decay=args.decay
            )
            output_name = 'transition_T_decay_multi.npz'
        else:
            T_sparse, _, row_base = generate_ppmi_future_transition_matrix(
                os.path.join(dataset_path, 'train'),
                code_num,
                threshold_value=args.threshold,
                topk=args.topk,
                future_steps=args.future_steps,
                positive_threshold=args.ppmi_positive_threshold
            )
            output_name = 'transition_T_ppmi_future.npz'

        print(f"Matrix shape: {T_sparse.shape}")
        print(f"Non-zero elements: {T_sparse.nnz}")
        print(f"Row base range: [{row_base.min() if len(row_base) > 0 else 0}, {row_base.max() if len(row_base) > 0 else 0}]")
        print(f"Transition value range: [{T_sparse.data.min() if T_sparse.nnz > 0 else 0:.4f}, {T_sparse.data.max() if T_sparse.nnz > 0 else 0:.4f}]")

        for split in splits:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                output_file = os.path.join(split_path, output_name)
                save_sparse_transition(output_file, T_sparse.toarray())
                print(f"Saved {variant} transition matrix to {output_file}")
