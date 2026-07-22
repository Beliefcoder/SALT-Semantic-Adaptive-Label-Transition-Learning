# SALT Paper Mainline Repository

This directory is a cleaned paper-facing version of the SALT mainline.
It keeps only the fixed paper configuration:

- dataset: `car_qwen25_hidden_v1_oof_candidate_seed50`
- empirical prior: `T_data`, time-decayed future transition matrix
- semantic prior: `T_sem`, Qwen label representation + supervised semantic projector + 3-fold OOF scoring
- model: LSTM encoder + dual-prior propagation + sample-adaptive fusion + TCRF + RIF
- training: seed `50`, batch size `32`, max epochs `40`

## Directory Layout

```text
data/car_qwen25_hidden_v1_oof_candidate_seed50/
  encoded/
  standard/train/
    transition_T_decay_multi.npz
    transition_T_llm.npz
  standard/valid/
  standard/test/

models/
  model.py
  layers.py

preprocess/
  generate_transition.py
  extract_llama_label_embeddings.py
  build_semantic_transition_advanced.py
  train_semantic_projector.py

train_salt.py
run_train.sh
configs/mainline_config.json
```

## Main Training

Default training does not cache the full dataset on GPU.

```bash
bash run_train.sh --gpu 0
```

Enable GPU dataset cache only when the GPU has enough free memory:

```bash
bash run_train.sh --gpu 0 --cache-dataset-on-gpu
```

Outputs are saved to:

```text
results/car_qwen25_hidden_v1_oof_candidate_seed50/
```

The printed evaluation line is intentionally compact:

```text
Validation Evaluation: loss ... f1_score ... micro_auprc ... top_k_precision ... top_k_recall ... seen_recall ...
```

`Not_Occurred` and unseen metrics are not printed or saved in this clean mainline entry.
Long-tail bucket output keeps only bucket F1 values.

## Prior Construction

The included dataset already contains the fixed priors:

```text
standard/train/transition_T_decay_multi.npz
standard/train/transition_T_llm.npz
```

To rebuild them:

```bash
MODEL_PATH=io/models/Qwen2.5-7B-Instruct bash scripts/build_priors.sh 0
```

Fixed prior parameters:

```text
T_data:
  future visit window K_w = 3
  decay gamma = 0.7
  source-frequency normalization
  row-wise top-k = 20

T_sem:
  candidate data top-k = 20
  candidate cosine top-k = 20
  random candidates per source = 4
  positive top-k = 20
  projector epochs = 10
  projector batch size = 512
  projector lr = 1e-3
  lambda_reg = 0.5
  OOF folds = 3
  semantic threshold = 0.05
  row-wise K_sem = 20
```

## Notes

This repository is intended for the paper mainline only. Ablation switches, exploratory branches, external baselines, binary-system tasks, and failed-run logs should remain outside this clean directory.
