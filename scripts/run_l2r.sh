#!/bin/bash
# =============================================================================
# L2R Baseline Experiment
# =============================================================================
# Following the original L2R paper experimental setup
# =============================================================================

set -e

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

DATASETS=${DATASETS:-"nq covidqa convqa newnewsqa"}
SAMPLES_PER_TASK=1000
TOP_K=5

# L2R specific settings (following original paper)
LR=1e-5
WARMUP_RATIO=0.1
MEMORY_SIZE=500  # replay memory size

BATCH_SIZE=1
MAX_STEPS=5000
EVAL_INTERVAL=500
SEED=42

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/baseline_l2r_${TIMESTAMP}"

echo "=============================================="
echo "L2R Baseline Experiment"
echo "=============================================="
echo "Datasets: ${DATASETS}"
echo "Memory size: ${MEMORY_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

python run.py \
    --datasets ${DATASETS} \
    --cl_method l2r \
    --retriever contriever \
    --gpu ${GPU} \
    --seed ${SEED} \
    --top_k ${TOP_K} \
    --lr ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --memory_size ${MEMORY_SIZE} \
    --samples_per_task ${SAMPLES_PER_TASK} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --eval_interval ${EVAL_INTERVAL} \
    --output_dir ${OUTPUT_DIR}

echo "=============================================="
echo "L2R baseline completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
