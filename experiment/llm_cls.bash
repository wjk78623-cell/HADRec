lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4  # 原始学习率
model_name_or_path="/mnt/data/833/wjk/huggyllama"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="/mnt/data/833/wjk/LEADER-pytorch-master0.5/data/mimic3/l_data"  # 填入数据集所在的文件夹路径
your_checkpopint_path="saved"  # 填入用来存储模型的路径
MAX_STEPS=5000  # 训练5000步，每200步保存checkpoint，训练后测试所有checkpoint找最佳
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="atc_hierarchy2"
MAX_SOURCE_LENGTH=2048

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

# Training Command - 使用 FP16 + DeepSpeed ZeRO-2（训练后验证策略）
# 每200步保存checkpoint，训练完成后测试所有checkpoint找最佳
deepspeed --num_gpus=1 --master_port $MASTER_PORT main_llm_cls.py \
    --deepspeed llm/ds.config \
    --do_train \
    --train_file $your_data_path/train_atc_hierarchy2.json \
    --test_file $your_data_path/test_atc_hierarchy2.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column atc_level_4 \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/lora-$date \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps ${MAX_STEPS} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 30\
    --logging_steps 100 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16 \
    --hierarchical_prediction \
    --balance_strategy combined \
    --use_cross_attention \
    --cross_threshold 0.1 \
    --drug_embedding_file /mnt/data/833/wjk/LEADER-pytorch-master0.5/data/mimic3/l_data/drug_embeddings.pt

# Testing Command
# Testing Command - 移除 deepspeed
CUDA_VISIBLE_DEVICES=0 python main_llm_cls.py \
    --do_predict \
    --train_file $your_data_path/train_atc_hierarchy2.json \
    --test_file $your_data_path/test_atc_hierarchy2.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column atc_level_4 \
    --model_name_or_path $model_name_or_path \
    --peft_path ${peft_path:-$your_checkpopint_path/lora-$date/checkpoint-$MAX_STEPS} \
    --output_dir results/$date \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_eval_batch_size 4 \
    --hierarchical_prediction \
    --use_cross_attention \
    --cross_threshold 0.1 \
    --drug_embedding_file /mnt/data/833/wjk/LEADER-pytorch-master0.5/data/mimic3/l_data/drug_embeddings.pt