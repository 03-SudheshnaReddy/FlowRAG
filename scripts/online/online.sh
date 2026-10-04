#!/bin/bash

METHODS="fp"
GPU=0
datasets="covidqa"
INDEX_TYPE="FLAT" 
USE_FAISS=true
embedding_model="contriever"

# 网格搜索参数
LEARNING_RATES="1e-3"
MAX_TRAINING_STEPS="3000"  # 设置一个较大的最大训练步数，每500步会自动评测

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
for embedding in $embedding_model
do
    echo ""
    echo "当前嵌入模型: ${embedding}"
    for method in $METHODS
    do
        echo ""
        echo "当前方法: ${method}" 
        for dataset in $datasets
        do
            # 创建输出目录
            output_dir="./output/online/${embedding}/${method}/${dataset}_t0.1"
            mkdir -p $output_dir
            echo ""
            echo "数据集: ${dataset}"
  
            for lr in $LEARNING_RATES
            do
                echo ""
                echo "学习率: ${lr}, 最大训练步数: ${MAX_TRAINING_STEPS}"
                
                echo "使用Faiss索引进行训练..."
                python run_manager.py \
                    --datasets ${dataset} \
                    --cl_retriever_method ${method} \
                    --retriever_name ${embedding} \
                    --index_type ${INDEX_TYPE} \
                    --gpu ${GPU} \
                    --lr ${lr} \
                    --step ${MAX_TRAINING_STEPS} \
                    --save_filename ./output/online/${embedding}/${method}/${dataset}_t0.1/lr_${lr}_maxsteps_${MAX_TRAINING_STEPS} \
                    --use_faiss \
                    --use_separate_index \
                    --temperature 0.1 \
                    --prompt_len 200 \
                    --prompt_layer 6 \
                    --use_ilf 1 \
                    --use_cef 0 \
                    --use_ggf 0
            
                # python test_manager.py \
                #     --datasets ${datasets} \
                #     --cl_retriever_method ${method} \
                #     --retriever_name ${embedding} \
                #     --index_type ${INDEX_TYPE} \
                #     --gpu ${GPU} \
                #     --lr ${lr} \
                #     --step ${MAX_TRAINING_STEPS} \
                #     --save_filename ./output/online/${embedding}/${method}/${dataset}_t0.1/lr_${lr}_maxsteps_${MAX_TRAINING_STEPS} \
                #     --use_faiss \
                #     --use_separate_index \
                #     --temperature 0.1
            done
        echo "训练完成，结果保存在 ${output_dir}/lr_${lr}_maxsteps_${MAX_TRAINING_STEPS} 中"
        echo "评测结果..."
        python grid_search_evaluator.py \
            --base_dir $output_dir \
            --output_dir ./grid_search_evaluation/${embedding}/${method}/${dataset}_t0.1
        done
    done
done
