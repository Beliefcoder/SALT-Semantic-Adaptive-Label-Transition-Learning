#!/usr/bin/env bash
set -euo pipefail

GPU=0
CACHE_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --cache-dataset-on-gpu)
      CACHE_FLAG="--cache-dataset-on-gpu"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

CUDA_VISIBLE_DEVICES="${GPU}" python train_salt.py \
  --gpu 0 \
  --dataset data \
  --seed 50 \
  --epochs 40 \
  --batch-size 32 \
  --result-tag salt_main_seed50 \
  ${CACHE_FLAG}
