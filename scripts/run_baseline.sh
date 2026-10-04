#!/bin/bash
# =============================================================================
# Baseline Methods - Fine-tuning Experiments
# =============================================================================
# Paper Settings (Baselines):
#   - Learning rate: 1e-5
#   - Warm-up ratio: 0.1
#   - Optimizer: AdamW
#   - Top-5 documents per query
#   - 1,000 QA pairs per dataset
# =============================================================================

set -e

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

DATASETS=${DATASETS:-"nq covidqa convqa newnewsqa"}
SAMPLES_PER_TASK=1000
TOP_K=5

# Baseline hyperparameters (from paper)
LR=1e-5
WARMUP_RATIO=0.1

BATCH_SIZE=1
MAX_STEPS=5000
EVAL_INTERVAL=500
SEED=42

# Method selection
METHOD=${1:-"replug"}  # replug, emdr, fid, offline

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/baseline_${METHOD}_${TIMESTAMP}"

echo "=============================================="
echo "Baseline Experiment: ${METHOD}"
echo "=============================================="
echo "Datasets: ${DATASETS}"
echo "Learning rate: ${LR}"
echo "Warm-up ratio: ${WARMUP_RATIO}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

python run.py \
    --datasets ${DATASETS} \
    --cl_method ${METHOD} \
    --retriever contriever \
    --gpu ${GPU} \
    --seed ${SEED} \
    --top_k ${TOP_K} \
    --lr ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --samples_per_task ${SAMPLES_PER_TASK} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --eval_interval ${EVAL_INTERVAL} \
    --output_dir ${OUTPUT_DIR}

echo "=============================================="
echo "Baseline ${METHOD} completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
