#!/bin/bash
# =============================================================================
# Run All Paper Experiments
# =============================================================================
# This script runs all experiments from the paper:
#   1. FlowRAG (proposed method)
#   2. Baselines: REPLUG, EMDR, FiD, Offline
#   3. L2R baseline
# =============================================================================

set -e

GPU=${GPU:-0}
DATASETS=${DATASETS:-"nq covidqa convqa newnewsqa"}

echo "=============================================="
echo "Running All Paper Experiments"
echo "=============================================="
echo "GPU: ${GPU}"
echo "Datasets: ${DATASETS}"
echo "=============================================="

# Create results directory
RESULTS_DIR="output/paper_experiments_$(date +%Y%m%d)"
mkdir -p ${RESULTS_DIR}

# 1. FlowRAG (Proposed Method)
echo ""
echo "[1/5] Running FlowRAG..."
GPU=${GPU} DATASETS="${DATASETS}" bash scripts/run_flowrag_paper.sh 2>&1 | tee ${RESULTS_DIR}/flowrag.log

# 2. REPLUG Baseline
echo ""
echo "[2/5] Running REPLUG baseline..."
GPU=${GPU} DATASETS="${DATASETS}" bash scripts/run_baseline.sh replug 2>&1 | tee ${RESULTS_DIR}/replug.log

# 3. EMDR Baseline
echo ""
echo "[3/5] Running EMDR baseline..."
GPU=${GPU} DATASETS="${DATASETS}" bash scripts/run_baseline.sh emdr 2>&1 | tee ${RESULTS_DIR}/emdr.log

# 4. FiD Baseline
echo ""
echo "[4/5] Running FiD baseline..."
GPU=${GPU} DATASETS="${DATASETS}" bash scripts/run_baseline.sh fid 2>&1 | tee ${RESULTS_DIR}/fid.log

# 5. Offline Baseline
echo ""
echo "[5/5] Running Offline baseline..."
GPU=${GPU} DATASETS="${DATASETS}" bash scripts/run_baseline.sh offline 2>&1 | tee ${RESULTS_DIR}/offline.log

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "=============================================="
