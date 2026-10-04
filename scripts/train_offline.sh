#!/bin/bash
# Offline Baseline Training Script
# Usage: bash scripts/train_offline.sh

set -e

DATASETS="nq"
RETRIEVER="contriever"
GPU=0

python run.py \
    --datasets ${DATASETS} \
    --cl_method offline \
    --retriever ${RETRIEVER} \
    --gpu ${GPU} \
    --max_steps 5000 \
    --eval_interval 500

echo "Offline training completed!"
