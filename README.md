# SALT: Semantic-Adaptive Label Transition Learning

This is the clean paper repository for the fixed SALT mainline.

## Contents

```text
data/data/                 processed dataset and fixed transition priors
models/                    SALT model implementation
preprocess/                data and prior construction scripts
train_salt.py              fixed mainline training entry
run_train.sh               simple training wrapper
scripts/build_priors.sh    rebuild T_data and T_sem
configs/mainline_config.json
```

## Main Training

Default setting: seed `50`, batch size `32`, max epochs `40`.
GPU dataset cache is off by default.

```bash
bash run_train.sh --gpu 0
```

Optional GPU cache:

```bash
bash run_train.sh --gpu 0 --cache-dataset-on-gpu
```

Results are saved to:

```text
results/data/
```

## Prior Construction

The released dataset already includes:

```text
data/data/standard/train/transition_T_decay_multi.npz
data/data/standard/train/transition_T_llm.npz
```

To rebuild the priors:

```bash
MODEL_PATH=io/models/Qwen2.5-7B-base bash scripts/build_priors.sh 0
```

Main prior settings:

```text
T_data: K_w=3, gamma=0.7, row-wise top-20
T_sem: Qwen2.5-7B base embeddings, 20/20/4 candidates, 3-fold OOF scoring,
       projector epochs=50, batch size=512, lr=1e-3, lambda_reg=0.5,
       threshold=0.05, row-wise K_sem=20
```
