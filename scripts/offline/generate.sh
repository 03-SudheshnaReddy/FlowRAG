#!/bin/bash

# 默认参数
METHODS="contriever"
# LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
# LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLM_MODEL="nvidia/Llama3-ChatQA-2-8B"
GPU=1
datasets="covidqa newnewsqa nq convqa"
INDEX_TYPE="FLAT" 
USE_FAISS=true

echo "=== Faiss索引离线评测脚本 ==="
echo "数据集: ${datasets}"
echo "检索方法: ${METHODS}"
echo "LLM模型: ${LLM_MODEL}"
echo "GPU: ${GPU}"
echo "索引类型: ${INDEX_TYPE}"
echo "使用Faiss: ${USE_FAISS}"
echo "================================"

for method in $METHODS
    do
        echo ""
        echo "处理检索方法: ${method}"
            
        echo "生成结果"
        python run_manager.py \
            --datasets ${datasets} \
            --cl_retriever_method "offline" \
            --retriever_name ${method} \
            --model_name_or_path ${LLM_MODEL} \
            --index_type ${INDEX_TYPE} \
            --gpu ${GPU} \
            --save_filename ./output/offline/${method}_sep_5 \
            --use_faiss \
            --retrieve_top_k 5 \
            --use_separate_index

        python test_manager.py \
            --datasets ${datasets} \
            --cl_retriever_method "offline" \
            --retriever_name ${method} \
            --model_name_or_path ${LLM_MODEL} \
            --index_type ${INDEX_TYPE} \
            --gpu ${GPU} \
            --save_filename ./output/offline/${method}_sep_5 \
            --use_faiss 
    done

echo ""
echo "=== 脚本执行完成 ==="
