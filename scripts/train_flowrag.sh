#!/bin/bash
# FlowRAG Training Script - Online Learning with FusionPrompt
# Usage: bash scripts/train_flowrag.sh

set -e

# Configuration
DATASETS="nq covidqa convqa newnewsqa"
CL_METHOD="fp"
RETRIEVER="contriever"
GPU=0

# FusionPrompt parameters
PROMPT_LEN=80
PROMPT_LAYER=4
USE_ILF="--use_ilf"
USE_CEF="--use_cef"

# Training parameters
LR=1e-4
BATCH_SIZE=1
MAX_STEPS=5000
EVAL_INTERVAL=500

# Output
OUTPUT_DIR="output/flowrag_$(date +%Y%m%d_%H%M%S)"

echo "Starting FlowRAG training..."
echo "Datasets: ${DATASETS}"
echo "Method: ${CL_METHOD}"
echo "Output: ${OUTPUT_DIR}"

python run.py \
    --datasets ${DATASETS} \
    --cl_method ${CL_METHOD} \
    --retriever ${RETRIEVER} \
    --gpu ${GPU} \
    --prompt_len ${PROMPT_LEN} \
    --prompt_layer ${PROMPT_LAYER} \
    ${USE_ILF} \
    ${USE_CEF} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --eval_interval ${EVAL_INTERVAL} \
    --output_dir ${OUTPUT_DIR}

echo "Training completed! Results saved to ${OUTPUT_DIR}"
