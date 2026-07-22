#!/usr/bin/env bash
set -euo pipefail

DATASET="data"
GPU="${1:-0}"
MODEL_PATH="${MODEL_PATH:-io/models/Qwen2.5-7B-Instruct}"

echo "dataset: ${DATASET}"
echo "This script rebuilds the two fixed SALT priors with the paper mainline parameters."

python preprocess/generate_transition.py \
  --dataset "${DATASET}" \
  --variant decay_multi \
  --future_steps 3 \
  --decay 0.7 \
  --threshold 0.0 \
  --topk 20

CUDA_VISIBLE_DEVICES="${GPU}" python preprocess/extract_llama_label_embeddings.py \
  --source_dataset "${DATASET}" \
  --variant_dataset "${DATASET}" \
  --model_path "${MODEL_PATH}" \
  --output_path "data/${DATASET}/artifacts/label_embeddings_qwen25_fp16.pt" \
  --batch_size 16 \
  --device cuda

CUDA_VISIBLE_DEVICES="${GPU}" python preprocess/build_semantic_transition_advanced.py \
  --dataset "${DATASET}" \
  --embedding_path "data/${DATASET}/artifacts/label_embeddings_qwen25_fp16.pt" \
  --transition_variant decay_multi \
  --positive_topk 20 \
  --candidate_data_topk 20 \
  --candidate_cosine_topk 20 \
  --candidate_random_per_row 4 \
  --epochs 10 \
  --batch_size 512 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --lambda_reg 0.5 \
  --oof_overlap \
  --oof_folds 3 \
  --output_threshold 0.05 \
  --output_topk 20 \
  --device cuda \
  --cosine_device cuda \
  --seed 50 \
  --output_path "data/${DATASET}/standard/train/transition_T_llm.npz" \
  --artifact_name "semantic_transition_oof_candidate_seed50"

python scripts/final_matrix_stats.py \
  --dataset "${DATASET}" \
  --result-root results
