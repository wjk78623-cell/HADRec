#!/bin/bash
# 完整流程：训练 → 测试 → 可视化
# 用法: bash experiment/full_pipeline.bash

set -e  # 遇到错误立即退出

echo ""
echo "========================================"
echo "   完整流程：训练 → 测试 → 可视化"
echo "========================================"
echo ""

# ============= 配置参数 =============
lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="/mnt/data/833/wjk/huggyllama"
your_data_path="/mnt/data/833/wjk/LEADER-pytorch-master0.5/data/mimic3/l_data"
your_checkpopint_path="saved"
MAX_STEPS=5000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="atc_hierarchy2"
MAX_SOURCE_LENGTH=2048

checkpoint_dir="${your_checkpopint_path}/lora-${date}"

# ============= Step 1: 训练（自动恢复）=============
echo "📝 Step 1/3: 训练模型"
echo "========================================"
echo ""

# 查找最新checkpoint
latest_checkpoint=$(ls -d ${checkpoint_dir}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$latest_checkpoint" ]; then
    echo "🆕 从头开始训练..."
    resume_arg=""
else
    step_num=$(basename $latest_checkpoint | sed 's/checkpoint-//')
    remaining=$((MAX_STEPS - step_num))
    
    if [ $remaining -le 0 ]; then
        echo "✅ 训练已完成 (${step_num}/${MAX_STEPS})，跳过训练"
        echo ""
    else
        echo "🔄 从 checkpoint-${step_num} 恢复训练 (剩余 ${remaining} 步)..."
        resume_arg="--resume_from_checkpoint ${latest_checkpoint} --ignore_data_skip"
    fi
fi

# 只有在训练未完成时才执行训练
final_checkpoint=$(ls -d ${checkpoint_dir}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
final_step=$(basename $final_checkpoint 2>/dev/null | sed 's/checkpoint-//')

if [ -z "$final_step" ] || [ $final_step -lt $MAX_STEPS ]; then
    echo "开始训练..."
    echo ""
    
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
        --max_source_length $MAX_SOURCE_LENGTH \
        --max_target_length 196 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --max_steps ${MAX_STEPS} \
        --eval_strategy no \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 30 \
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
        --drug_embedding_file /mnt/data/833/wjk/LEADER-pytorch-master0.5/data/mimic3/l_data/drug_embeddings.pt \
        $resume_arg
    
    echo ""
    echo "✅ 训练完成"
    echo ""
fi

# ============= Step 2: 测试所有checkpoints =============
echo ""
echo "📝 Step 2/3: 测试所有checkpoints"
echo "========================================"
echo ""

bash experiment/test_all_checkpoints.bash

echo ""
echo "✅ 测试完成"
echo ""

# ============= Step 3: 生成可视化 =============
echo ""
echo "📝 Step 3/3: 生成可视化"
echo "========================================"
echo ""

# 找出最佳checkpoint
best_checkpoint=$(tail -n +2 checkpoint_comparison.csv | sort -t',' -k13 -rn | head -n1 | cut -d',' -f1)

if [ -n "$best_checkpoint" ]; then
    echo "🏆 最佳checkpoint: ${best_checkpoint}"
    echo ""
    
    # 生成可视化
    if [ -f "results/${best_checkpoint}/test_predictions.json" ]; then
        echo "生成真实注意力可视化..."
        python visualize_real_attention.py results/${best_checkpoint}/test_predictions.json
        
        echo ""
        echo "生成深度分析图表..."
        python advanced_visualization.py
        
        echo ""
        echo "✅ 可视化完成"
    else
        echo "⚠️  未找到预测文件: results/${best_checkpoint}/test_predictions.json"
    fi
else
    echo "⚠️  未找到最佳checkpoint"
fi

# ============= 最终总结 =============
echo ""
echo "========================================"
echo "   🎉 所有任务完成！"
echo "========================================"
echo ""

echo "📊 结果文件:"
echo "   - checkpoint_comparison.csv (所有checkpoint性能)"
echo "   - checkpoint_comparison.txt (详细报告)"
echo ""

echo "📈 可视化图表:"
echo "   - visualizations/real_attention_sample*.png (真实注意力热图)"
echo "   - visualizations/real_attention_entropy.png (注意力熵值分布)"
echo "   - visualizations/fig1-6_*.png (深度分析图表)"
echo ""

echo "💡 下一步:"
echo "   1. 查看最佳模型: cat checkpoint_comparison.txt | grep '最佳'"
echo "   2. 查看图表: ls visualizations/*.png"
echo "   3. 分析结果: python -c 'import pandas as pd; df=pd.read_csv(\"checkpoint_comparison.csv\"); print(df.sort_values(\"L4_Jaccard\", ascending=False).head())'"
echo ""

echo "🎯 快速查看:"
if [ -n "$best_checkpoint" ]; then
    echo "   最佳checkpoint: ${best_checkpoint}"
    best_jaccard=$(grep "^${best_checkpoint}," checkpoint_comparison.csv | cut -d',' -f13)
    echo "   L4 Jaccard: ${best_jaccard}"
fi
echo ""
