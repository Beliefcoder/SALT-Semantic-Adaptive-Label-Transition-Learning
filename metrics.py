import csv
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


TOP_KS = [3, 5, 10, 15, 20, 25, 30, 35]


def format_metric(value):
    if value is None:
        return 'nan'
    return f'{value:.4f}'


def ensure_csv_header(output_path, headers):
    if not os.path.isfile(output_path):
        return False

    with open(output_path, 'r', newline='') as f:
        reader = csv.reader(f)
        existing_headers = next(reader, [])

    if existing_headers == headers:
        return True

    backup_path = f'{output_path}.legacy'
    backup_index = 1
    while os.path.exists(backup_path):
        backup_path = f'{output_path}.legacy{backup_index}'
        backup_index += 1

    os.replace(output_path, backup_path)
    print(f'\r    Existing CSV header mismatch, backed up old file to {backup_path}')
    return False


def build_binary_predictions(y_true_hot, y_pred, label_lookup=None):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = int(np.sum(y_true_hot[i] == 1))
        if true_number <= 0:
            continue

        selected = 0
        for label in y_pred[i]:
            local_label = label if label_lookup is None else label_lookup[label]
            if local_label < 0:
                continue
            result[i][local_label] = 1
            selected += 1
            if selected >= true_number:
                break
    return result


def f1(y_true_hot, y_pred, metrics='weighted'):
    result = build_binary_predictions(y_true_hot, y_pred)
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    valid_count = 0
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        if len(t) == 0:
            continue
        valid_count += 1
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)
    if valid_count == 0:
        return a, r
    return a / valid_count, r / valid_count


def safe_metric(metric_fn, *args, **kwargs):
    try:
        return metric_fn(*args, **kwargs)
    except ValueError:
        return None


def compute_multilabel_score_metrics(y_true_hot, y_score):
    valid_macro_mask = np.logical_and(np.sum(y_true_hot, axis=0) > 0,
                                      np.sum(y_true_hot, axis=0) < y_true_hot.shape[0])

    metrics = {
        'micro_auroc': safe_metric(roc_auc_score, y_true_hot, y_score, average='micro'),
        'micro_auprc': safe_metric(average_precision_score, y_true_hot, y_score, average='micro'),
        'macro_auroc': None,
        'macro_auprc': None,
    }

    if np.any(valid_macro_mask):
        y_true_macro = y_true_hot[:, valid_macro_mask]
        y_score_macro = y_score[:, valid_macro_mask]
        metrics['macro_auroc'] = safe_metric(roc_auc_score, y_true_macro, y_score_macro, average='macro')
        metrics['macro_auprc'] = safe_metric(average_precision_score, y_true_macro, y_score_macro, average='macro')

    return metrics


def calculate_occurred(historical, y, preds, ks):
    r1 = np.zeros((len(ks), ))
    r2 = np.zeros((len(ks),))
    n = np.sum(y, axis=-1)
    valid_mask = n > 0
    if not np.any(valid_mask):
        return r1, r2

    for i, k in enumerate(ks):
        pred_k = np.zeros_like(y)
        for t in range(len(pred_k)):
            pred_k[t][preds[t][:k]] = 1
        pred_occurred = np.logical_and(historical, pred_k)
        pred_not_occurred = np.logical_and(np.logical_not(historical), pred_k)
        pred_occurred_true = np.logical_and(pred_occurred, y)
        pred_not_occurred_true = np.logical_and(pred_not_occurred, y)

        n_valid = n[valid_mask]
        r1[i] = np.mean(np.sum(pred_occurred_true[valid_mask], axis=-1) / n_valid)
        r2[i] = np.mean(np.sum(pred_not_occurred_true[valid_mask], axis=-1) / n_valid)
    return r1, r2


def calculate_history_decomposition(last_history, all_history, y, preds, ks):
    repeat_last = np.zeros((len(ks),))
    history_shift = np.zeros((len(ks),))
    emerging_never = np.zeros((len(ks),))
    n = np.sum(y, axis=-1)
    valid_mask = n > 0
    if not np.any(valid_mask):
        return repeat_last, history_shift, emerging_never

    last_history = last_history.astype(bool)
    all_history = all_history.astype(bool)
    y = y.astype(bool)
    earlier_history = np.logical_and(all_history, np.logical_not(last_history))
    never_history = np.logical_not(all_history)

    for i, k in enumerate(ks):
        pred_k = np.zeros_like(y, dtype=bool)
        for t in range(len(pred_k)):
            pred_k[t][preds[t][:k]] = True
        repeat_true = np.logical_and(np.logical_and(last_history, pred_k), y)
        shift_true = np.logical_and(np.logical_and(earlier_history, pred_k), y)
        emerging_true = np.logical_and(np.logical_and(never_history, pred_k), y)

        n_valid = n[valid_mask]
        repeat_last[i] = np.mean(np.sum(repeat_true[valid_mask], axis=-1) / n_valid)
        history_shift[i] = np.mean(np.sum(shift_true[valid_mask], axis=-1) / n_valid)
        emerging_never[i] = np.mean(np.sum(emerging_true[valid_mask], axis=-1) / n_valid)
    return repeat_last, history_shift, emerging_never


def calculate_global_history_occurrence(all_history, y, preds, ks):
    global_occurred = np.zeros((len(ks),))
    global_not_occurred = np.zeros((len(ks),))
    all_history = all_history.astype(bool)
    y = y.astype(bool)
    total_targets = float(np.sum(y))
    if total_targets <= 0:
        return global_occurred, global_not_occurred

    seen_targets = np.logical_and(y, all_history)
    unseen_targets = np.logical_and(y, np.logical_not(all_history))
    for i, k in enumerate(ks):
        pred_k = np.zeros_like(y, dtype=bool)
        for row in range(len(pred_k)):
            pred_k[row][preds[row][:k]] = True
        global_occurred[i] = np.sum(np.logical_and(pred_k, seen_targets)) / total_targets
        global_not_occurred[i] = np.sum(np.logical_and(pred_k, unseen_targets)) / total_targets
    return global_occurred, global_not_occurred


def calculate_seen_unseen_group_metrics(all_history, y, preds, ks):
    seen_recall = np.zeros((len(ks),))
    unseen_recall = np.zeros((len(ks),))
    unseen_hit = np.zeros((len(ks),))
    all_history = all_history.astype(bool)
    y = y.astype(bool)

    seen_targets = np.logical_and(y, all_history)
    unseen_targets = np.logical_and(y, np.logical_not(all_history))
    seen_total = float(np.sum(seen_targets))
    unseen_total = float(np.sum(unseen_targets))
    unseen_sample_mask = np.sum(unseen_targets, axis=1) > 0
    unseen_sample_count = float(np.sum(unseen_sample_mask))

    for i, k in enumerate(ks):
        pred_k = np.zeros_like(y, dtype=bool)
        for row in range(len(pred_k)):
            pred_k[row][preds[row][:k]] = True

        seen_hits = np.sum(np.logical_and(pred_k, seen_targets))
        unseen_hits = np.sum(np.logical_and(pred_k, unseen_targets))
        seen_recall[i] = seen_hits / seen_total if seen_total > 0 else 0.0
        unseen_recall[i] = unseen_hits / unseen_total if unseen_total > 0 else 0.0

        if unseen_sample_count > 0:
            sample_has_unseen_hit = np.sum(np.logical_and(pred_k, unseen_targets), axis=1) > 0
            unseen_hit[i] = np.sum(np.logical_and(sample_has_unseen_hit, unseen_sample_mask)) / unseen_sample_count
    return seen_recall, unseen_recall, unseen_hit


def compute_bucket_recall(y_true_hot, y_pred, ks, bucket_mask):
    recall = np.zeros((len(ks),))
    valid_count = 0
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(np.logical_and(true_hot == 1, bucket_mask))[0].tolist()
        t = set(true)
        if len(t) == 0:
            continue
        valid_count += 1
        for i, k in enumerate(ks):
            p = set(pred[:k])
            recall[i] += len(p.intersection(t)) / len(t)

    if valid_count == 0:
        return recall
    return recall / valid_count


def compute_bucket_f1(y_true_hot, y_pred, bucket_mask):
    bucket_indices = np.where(bucket_mask)[0]
    if len(bucket_indices) == 0:
        return None

    y_true_bucket = y_true_hot[:, bucket_mask]
    if np.sum(y_true_bucket) == 0:
        return None

    label_lookup = np.full(bucket_mask.shape[0], -1, dtype=np.int32)
    label_lookup[bucket_indices] = np.arange(len(bucket_indices), dtype=np.int32)
    y_pred_bucket = build_binary_predictions(y_true_bucket, y_pred, label_lookup=label_lookup)
    return f1_score(y_true=y_true_bucket, y_pred=y_pred_bucket, average='weighted', zero_division=0)


def compute_long_tail_metrics(y_true_hot, y_pred, train_label_freq, ks):
    if train_label_freq is None:
        return {}

    train_label_freq = np.asarray(train_label_freq).reshape(-1)
    bucket_masks = {
        'LE5': train_label_freq <= 5,
        'FREQ6TO20': np.logical_and(train_label_freq >= 6, train_label_freq <= 20),
        'GT20': train_label_freq > 20,
    }

    bucket_metrics = {}
    for bucket_name, bucket_mask in bucket_masks.items():
        bucket_metrics[bucket_name] = {
            'f1': compute_bucket_f1(y_true_hot, y_pred, bucket_mask),
            'recall': compute_bucket_recall(y_true_hot, y_pred, ks, bucket_mask),
        }
    return bucket_metrics


def evaluate_codes(code_adj, model, dataset, loss_fn, output_size, historical=None, epoch=0,
                   save_csv=True, csv_path='result/evaluation_results.csv', train_label_freq=None,
                   split_name='Validation', all_historical=None):
    del code_adj
    model.eval()
    total_loss = 0.0
    labels = dataset.label()
    scores = []
    preds = []
    for step in range(len(dataset)):
        code_x, visit_lens, divided, y, neighbors = dataset[step]
        del divided, neighbors
        output = model(code_x, visit_lens)
        prob = torch.sigmoid(output)
        pred = torch.argsort(prob, dim=-1, descending=True)
        scores.append(prob.detach().cpu().numpy())
        preds.append(pred)
        loss = loss_fn(output, y)
        total_loss += loss.item() * output_size * len(code_x)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    avg_loss = total_loss / dataset.size()
    scores = np.vstack(scores)
    preds = torch.vstack(preds).detach().cpu().numpy()

    f1_score_val = f1(labels, preds)
    multilabel_metrics = compute_multilabel_score_metrics(labels, scores)
    precision, recall = top_k_prec_recall(labels, preds, ks=TOP_KS)
    long_tail_metrics = compute_long_tail_metrics(labels, preds, train_label_freq, ks=TOP_KS)

    r1, r2 = None, None
    history_decomposition = None
    global_history_occurrence = None
    seen_unseen_group_metrics = None
    if historical is not None:
        r1, r2 = calculate_occurred(historical, labels, preds, ks=TOP_KS)
        if all_historical is not None:
            history_decomposition = calculate_history_decomposition(historical, all_historical, labels, preds, ks=TOP_KS)
            global_history_occurrence = calculate_global_history_occurrence(all_historical, labels, preds, ks=TOP_KS)
            seen_unseen_group_metrics = calculate_seen_unseen_group_metrics(all_historical, labels, preds, ks=TOP_KS)
        score_sum = f1_score_val + np.sum(recall)
        seen_recall_print = seen_unseen_group_metrics[0] if seen_unseen_group_metrics is not None else r1
        print('\r    %s Evaluation: loss: %.4f ---- Sum: %.4f --- f1_score: %.4f --- micro_auroc: %s --- micro_auprc: %s --- macro_auroc: %s --- macro_auprc: %s --- top_k_precision: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f --- seen_recall: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f'
              % (split_name, avg_loss, score_sum, f1_score_val,
                 format_metric(multilabel_metrics['micro_auroc']),
                 format_metric(multilabel_metrics['micro_auprc']),
                 format_metric(multilabel_metrics['macro_auroc']),
                 format_metric(multilabel_metrics['macro_auprc']),
                 precision[0], precision[1], precision[2], precision[3], precision[4], precision[5], precision[6], precision[7],
                 recall[0], recall[1], recall[2], recall[3], recall[4], recall[5], recall[6], recall[7],
                 seen_recall_print[0], seen_recall_print[1], seen_recall_print[2], seen_recall_print[3],
                 seen_recall_print[4], seen_recall_print[5], seen_recall_print[6], seen_recall_print[7]))
    else:
        print('\r    %s Evaluation: loss: %.4f --- f1_score: %.4f --- micro_auroc: %s --- micro_auprc: %s --- macro_auroc: %s --- macro_auprc: %s --- top_k_precision: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f'
              % (split_name, avg_loss, f1_score_val,
                 format_metric(multilabel_metrics['micro_auroc']),
                 format_metric(multilabel_metrics['micro_auprc']),
                 format_metric(multilabel_metrics['macro_auroc']),
                 format_metric(multilabel_metrics['macro_auprc']),
                 precision[0], precision[1], precision[2], precision[3], precision[4], precision[5], precision[6], precision[7],
                 recall[0], recall[1], recall[2], recall[3], recall[4], recall[5], recall[6], recall[7]))

    if long_tail_metrics:
        bucket_summaries = []
        for bucket_name, bucket_metric in long_tail_metrics.items():
            bucket_summaries.append(
                f'{bucket_name}(f1={format_metric(bucket_metric["f1"])})'
            )
        print(f'    Long-tail buckets: {" | ".join(bucket_summaries)}')

    if save_csv:
        save_evaluation_to_csv(
            epoch,
            avg_loss,
            f1_score_val,
            precision,
            recall,
            r1,
            r2,
            multilabel_metrics=multilabel_metrics,
            long_tail_metrics=long_tail_metrics,
            history_decomposition=history_decomposition,
            global_history_occurrence=global_history_occurrence,
            seen_unseen_group_metrics=seen_unseen_group_metrics,
            output_path=csv_path,
            loss_header=f'{split_name}_Loss',
        )

    return avg_loss, f1_score_val


def evaluate_hf(model, dataset, loss_fn, output_size=1, historical=None, epoch=0,
                save_csv=False, csv_path='result/evaluation_results_hf.csv', split_name='Validation',
                all_historical=None):
    del all_historical
    del historical
    model.eval()
    total_loss = 0.0
    labels = dataset.label()
    outputs = []
    preds = []
    for step in range(len(dataset)):
        code_x, visit_lens, divided, y, neighbors = dataset[step]
        del divided, neighbors
        output = model(code_x, visit_lens).squeeze()
        loss = loss_fn(output, y)
        total_loss += loss.item() * output_size * len(code_x)
        output = output.detach().cpu().numpy()
        outputs.append(output)
        pred = (output > 0).astype(int)
        preds.append(pred)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    avg_loss = total_loss / dataset.size()
    outputs = np.concatenate(outputs)
    preds = np.concatenate(preds)
    auc = roc_auc_score(labels, outputs)
    f1_score_ = f1_score(labels, preds)
    print('\r    %s Evaluation: loss: %.4f --- auc: %.4f --- f1_score: %.4f' % (split_name, avg_loss, auc, f1_score_))
    if save_csv:
        save_hf_evaluation_to_csv(epoch, avg_loss, auc, f1_score_, output_path=csv_path,
                                  loss_header=f'{split_name}_Loss')
    return avg_loss, f1_score_


def save_evaluation_to_csv(epoch, avg_loss, f1_score_val, precision, recall, r1, r2,
                           multilabel_metrics=None, long_tail_metrics=None,
                           history_decomposition=None,
                           global_history_occurrence=None,
                           seen_unseen_group_metrics=None,
                           output_path='result/evaluation_results.csv', loss_header='Validation_Loss'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    headers = ['Epoch', loss_header, 'F1_Score',
               'Micro_AUROC', 'Micro_AUPRC', 'Macro_AUROC', 'Macro_AUPRC'] + \
              [f'Precision_Top{k}' for k in TOP_KS] + \
              [f'Recall_Top{k}' for k in TOP_KS]
    if seen_unseen_group_metrics is not None:
        headers += [f'SeenRecall_Top{k}' for k in TOP_KS]
    elif r1 is not None:
        headers += [f'SeenRecall_Top{k}' for k in TOP_KS]

    if long_tail_metrics:
        for bucket_name in long_tail_metrics:
            headers.append(f'Bucket_{bucket_name}_F1')

    file_exists = ensure_csv_header(output_path, headers)

    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)

        row = [
            epoch + 1,
            avg_loss,
            f1_score_val,
            '' if multilabel_metrics is None or multilabel_metrics['micro_auroc'] is None else multilabel_metrics['micro_auroc'],
            '' if multilabel_metrics is None or multilabel_metrics['micro_auprc'] is None else multilabel_metrics['micro_auprc'],
            '' if multilabel_metrics is None or multilabel_metrics['macro_auroc'] is None else multilabel_metrics['macro_auroc'],
            '' if multilabel_metrics is None or multilabel_metrics['macro_auprc'] is None else multilabel_metrics['macro_auprc'],
        ]
        row.extend(precision.tolist())
        row.extend(recall.tolist())
        if seen_unseen_group_metrics is not None:
            seen_recall, _, _ = seen_unseen_group_metrics
            row.extend(seen_recall.tolist())
        elif r1 is not None:
            row.extend(r1.tolist())

        if long_tail_metrics:
            for bucket_metric in long_tail_metrics.values():
                row.append('' if bucket_metric['f1'] is None else bucket_metric['f1'])

        writer.writerow(row)

    print(f'\r    Evaluation results saved to {output_path}')


def save_hf_evaluation_to_csv(epoch, avg_loss, auc, f1_score_val, output_path='result/evaluation_results_hf.csv',
                              loss_header='Validation_Loss'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    headers = ['Epoch', loss_header, 'AUC', 'F1_Score']
    file_exists = ensure_csv_header(output_path, headers)

    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([epoch + 1, avg_loss, auc, f1_score_val])

    print(f'\r    Evaluation results saved to {output_path}')
