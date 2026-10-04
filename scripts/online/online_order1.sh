#!/bin/bash

METHODS="replug"
GPU=1
datasets="covidqa newnewsqa convqa nq"
INDEX_TYPE="FLAT" 
USE_FAISS=true

# 网格搜索参数
LEARNING_RATES="1e-5"
MAX_TRAINING_STEPS="500"  # 设置一个较大的最大训练步数，每500步会自动评测

echo "=== Faiss索引在线评测网格搜索脚本 ==="
echo "数据集: ${datasets}"
echo "方法: ${METHODS}"
echo "GPU: ${GPU}"
echo "索引类型: ${INDEX_TYPE}"
echo "使用Faiss: ${USE_FAISS}"
echo "学习率范围: ${LEARNING_RATES}"
echo "最大训练步数: ${MAX_TRAINING_STEPS}"
echo "评测间隔: 每500步"
echo "================================"

for method in $METHODS
do
    echo ""
    echo "处理方法: ${method}"
    
    for lr in $LEARNING_RATES
    do
        echo ""
        echo "学习率: ${lr}, 最大训练步数: ${MAX_TRAINING_STEPS}"
        
        echo "使用Faiss索引进行训练..."
        python run_manager.py \
            --datasets ${datasets} \
            --cl_retriever_method $method \
            --retriever_name bge \
            --index_type ${INDEX_TYPE} \
            --gpu ${GPU} \
            --lr ${lr} \
            --step ${MAX_TRAINING_STEPS} \
            --save_filename ./output/online/order1/bge/${method}_order1_500_t0.1/lr_${lr}/ \
            --use_faiss \
            --use_separate_index \
            --temperature 0.1
        # python test_manager.py \
        #     --datasets ${datasets} \
        #     --cl_retriever_method $method \
        #     --retriever_name contriever \
        #     --index_type ${INDEX_TYPE} \
        #     --gpu ${GPU} \
        #     --lr ${lr} \
        #     --step ${MAX_TRAINING_STEPS} \
        #     --save_filename ./output/online/order1/bge/${method}_order1_500_t0.1/lr_${lr}/ \
        #     --use_faiss \
        #     --use_separate_index \
        #     --temperature 0.1
    done
done

echo ""
echo "=== 脚本执行完成 ==="
