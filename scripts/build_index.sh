#!/bin/bash
# Build Faiss Index Script
# Usage: bash scripts/build_index.sh [retriever]

set -e

RETRIEVER=${1:-contriever}
DATASETS="nq covidqa convqa newnewsqa"

echo "Building Faiss index with ${RETRIEVER}..."

for dataset in ${DATASETS}; do
    echo "Processing ${dataset}..."
    python -m src.retrieval.index \
        --dataset ${dataset} \
        --retriever ${RETRIEVER} \
        --index_type FLAT
done

echo "Index building completed!"
