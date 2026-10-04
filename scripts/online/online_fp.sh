TEMPERATURES="0.1"
LRS="1e-4"
Pro_len="80"
Pompt_layer="4"
cef_t="1"
datasets="multirc"
steps="5000 6000 7000 8000 9000 10000"
for t in $TEMPERATURES
do
   for lr in $LRS
   do
      for prompt_len in $Pro_len
      do
        for prompt_layer in $Pompt_layer
        do
        # for dataset in $datasets
        #   do
          for cef_t in $cef_t
          do
            for step in $steps
            do
            python run_manager.py  --dataset ${datasets} --cl_retriever_method 'fp' --retriever_name e5 --lr ${lr} --gpu  0 --temperature ${t} --prompt_len ${prompt_len} --prompt_layer  ${prompt_layer} --use_ilf 1 --use_cef  1 --use_ggf 0  --step ${step} --save_filename ./new_output_cef_step25/${dataset}/fp_lr${lr}_t${t}_layer${prompt_layer}_len${prompt_len}_step${step}
            python test_manager.py  --dataset ${datasets} --cl_retriever_method 'fp' --retriever_name e5 --lr ${lr} --gpu  1 --temperature ${t} --prompt_len ${prompt_len} --prompt_layer  ${prompt_layer} --use_ilf 1 --use_cef  1 --use_ggf 0  --step ${step} --save_filename ./new_output_cef_step25/${dataset}/fp_lr${lr}_t${t}_layer${prompt_layer}_len${prompt_len}_step${step}
            done
          done
          # done
        done
      done
   done
done