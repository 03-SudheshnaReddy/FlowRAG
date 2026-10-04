#!/bin/bash

METHODS="e5"
GPU=0
datasets="covidqa newnewsqa nq convqa"
INDEX_TYPE="FLAT" 
USE_FAISS=true
USE_INCREMENTAL=false  # 是否使用增量向量数据库
USE_SEPARATE=false     # 是否使用单独索引（每个数据集只用自己的知识库）

# 检查是否有 --incremental 参数
if [[ "$*" == *"--incremental"* ]]; then
    USE_INCREMENTAL=true
fi

# 检查是否有 --separate 参数
if [[ "$*" == *"--separate"* ]]; then
    USE_SEPARATE=true
fi

echo "=== Faiss索引离线评测脚本 ==="
echo "数据集: ${datasets}"
echo "方法: ${METHODS}"
echo "GPU: ${GPU}"
echo "索引类型: ${INDEX_TYPE}"
echo "使用Faiss: ${USE_FAISS}"
echo "增量模式: ${USE_INCREMENTAL}"
echo "单独索引模式: ${USE_SEPARATE}"
echo "================================"

for method in $METHODS
    do
        echo ""
        echo "处理方法: ${method}"
            
        echo "步骤2: 使用Faiss索引进行训练..."
        cmd_args="--datasets ${datasets} \
            --datasets ${datasets} \
            --cl_retriever_method "offline" \
            --retriever_name ${method} \
            --model_name_or_path ${LLM_MODEL} \
            --index_type ${INDEX_TYPE} \
            --gpu ${GPU} \
            --save_filename ./output/offline/${method}_sep \
            --use_faiss \
            --prompter_type "mistral"
        
        # if [ "$USE_INCREMENTAL" = true ]; then
        #     cmd_args="${cmd_args} --use_incremental_index"
        # fi
        
        # if [ "$USE_SEPARATE" = true ]; then
        cmd_args="${cmd_args} --use_separate_index"
        # fi
        
        echo "执行命令: python run_manager.py ${cmd_args}"
        python run_manager.py ${cmd_args}
    done

echo ""
echo "=== 脚本执行完成 ==="
