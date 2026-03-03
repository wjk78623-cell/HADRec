
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import numpy as np

# 修复PyTorch 2.6的weights_only问题
# 使用torch.serialization.add_safe_globals
import torch.serialization
# 添加所有numpy相关类型
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.core.multiarray.scalar,
])
# 添加numpy.dtypes相关类型（字符串形式）
try:
    import numpy.dtypes
    torch.serialization.add_safe_globals([
        numpy.dtypes.UInt32DType,
        numpy.dtypes.Int64DType,
        numpy.dtypes.Float64DType,
    ])
except (ImportError, AttributeError):
    pass
from torch.utils.data import DataLoader
from datasets import load_dataset

from llm.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, HfArgumentParser, Seq2SeqTrainingArguments
from transformers import AutoModel, AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# 导入Qwen模型
from llm.llama import LlamaForMedRec
from llm.qwen import QwenForMedRec
from llm.trainer_seq2seq import MedRecTrainer
from llm.lora_cls import PeftModelForCLS
from llm.arguments import DataTrainingArguments, ModelArguments
from llm.data_processor.llama import llama_train_cls, llama_eval_cls, apply_balancing_strategy
from llm.data_processor.collator import LongestSequenceCollator
from generators.data import Voc, EHRTokenizer
from evaluate import evaluate_jsonlines
import time


class HierarchicalMedRecTrainer(Trainer):
    def __init__(self, *args, hierarchical_prediction=True, **kwargs):
        self.hierarchical_prediction = hierarchical_prediction
        super().__init__(*args, **kwargs)
        # 用于存储预测时的注意力权重
        self.attention_weights_cache = []
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """重写evaluate方法，在验证前清理显存"""
        import gc
        
        # 验证前清理显存
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 验证前清理显存完成")
        
        # 调用父类的evaluate方法
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 提取层级标签
        labels_l1 = inputs.pop("labels_l1", None)
        labels_l2 = inputs.pop("labels_l2", None)
        labels_l3 = inputs.pop("labels_l3", None)
        labels_l4 = inputs.pop("labels_l4", None)
        
        # 提取层级掩码
        mask_l2 = inputs.pop("mask_l2", None)
        mask_l3 = inputs.pop("mask_l3", None)
        mask_l4 = inputs.pop("mask_l4", None)
        

        
        # 调用模型前向传播
        if self.hierarchical_prediction:
            outputs = model(
                **inputs,
                labels_l1=labels_l1,
                labels_l2=labels_l2,
                labels_l3=labels_l3,
                labels_l4=labels_l4,
                mask_l2=mask_l2,
                mask_l3=mask_l3,
                mask_l4=mask_l4
              
            )
        else:
            outputs = model(
                **inputs,
                labels_l4=labels_l4,
            )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # 重写预测步骤以处理层级信息
        inputs = self._prepare_inputs(inputs)

        # 提取层级标签和掩码
        labels_l1 = inputs.pop("labels_l1", None)
        labels_l2 = inputs.pop("labels_l2", None)
        labels_l3 = inputs.pop("labels_l3", None)
        labels_l4 = inputs.pop("labels_l4", None)
        
        mask_l2 = inputs.pop("mask_l2", None)
        mask_l3 = inputs.pop("mask_l3", None)
        mask_l4 = inputs.pop("mask_l4", None)
        

        # 检查是否有层级标签
        has_labels = any(x is not None for x in [labels_l1, labels_l2, labels_l3, labels_l4])


        # 调用模型
        if self.hierarchical_prediction:
            outputs = model(
                **inputs,
                labels_l1=labels_l1,
                labels_l2=labels_l2,
                labels_l3=labels_l3,
                labels_l4=labels_l4,
                mask_l2=mask_l2,
                mask_l3=mask_l3,
                mask_l4=mask_l4
            )
        else:
            outputs = model(
                **inputs,
                labels_l4=labels_l4,
            )

        if ignore_keys is None:
            ignore_keys = []

        with torch.no_grad():
            # 保存注意力权重到缓存
            # 注意：模型返回的注意力权重在 outputs.attentions 字段中
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # 将注意力权重转移到CPU并转换为numpy
                attn_weights = {}
                for key, value in outputs.attentions.items():
                    if value is not None:
                        attn_weights[key] = value.cpu().numpy()
                self.attention_weights_cache.append(attn_weights)
                # 调试信息：只在第一个样本时打印
                if len(self.attention_weights_cache) == 1:
                    print(f"✅ 成功捕获注意力权重: {list(attn_weights.keys())}")
            else:
                self.attention_weights_cache.append(None)
                # 调试信息：只在第一个样本时打印
                if len(self.attention_weights_cache) == 1:
                    print(f"⚠️ 未捕获到注意力权重 - hasattr: {hasattr(outputs, 'attentions')}, is None: {outputs.attentions is None if hasattr(outputs, 'attentions') else 'N/A'}")
            
            if has_labels:
                loss = outputs.loss
                loss = loss.mean().detach()
                if self.hierarchical_prediction:
                    return (loss, outputs.logits, (labels_l1, labels_l2, labels_l3, labels_l4))
                else:
                    logits = outputs.logits
                    if isinstance(logits, tuple):
                        logits = logits[-1]
                    return (loss, logits, labels_l4)
            else:
                loss = None
                logits = outputs.logits
                if not self.hierarchical_prediction and isinstance(logits, tuple):
                    logits = logits[-1]
                return (loss, logits, None)


# save model for PeftModel
class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        if state.is_world_process_zero:
            print(f'+++++++++++++++++save call back (step {state.global_step})++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            
            # 🔥 只保存PEFT adapter权重（轻量级，几十MB）
            # 不保存DeepSpeed的完整训练状态（12GB+）
            print(f"💾 保存LoRA adapter到: {checkpoint_folder}")
            kwargs["model"].save_pretrained(checkpoint_folder)

            # 清理可能保存的完整模型文件
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
                print(f"🧹 已删除完整模型文件: pytorch_model.bin")
            
            print(f"✅ LoRA权重保存完成")
            return control




def train():
    # ============= 🎲 可复现性设置（增强版） =============
    # 设置全局随机种子，确保每次运行结果一致
    import random
    SEED = 42
    
    # 1. 设置环境变量（必须在其他设置之前）
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA确定性操作
    
    # 2. 设置Python随机种子
    random.seed(SEED)
    
    # 3. 设置NumPy随机种子
    np.random.seed(SEED)
    
    # 4. 设置PyTorch随机种子
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # 5. CUDA确定性设置（可能会稍微降低性能，但保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ 全局随机种子已设置: {SEED}")
    print(f"  - Python random: {SEED}")
    print(f"  - NumPy: {SEED}")
    print(f"  - PyTorch: {SEED}")
    print(f"  - CUDA: 确定性模式")
    print(f"  - CUBLAS: 确定性配置")
    # ====================================================
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    device_map = "auto"
    
    # 内存优化设置
    training_args.dataloader_pin_memory = False  # 禁用pin_memory节省内存
    training_args.dataloader_num_workers = 0     # 减少数据加载进程
    
    # 调试：打印所有参数
    print("=== 参数调试信息 ===")
    print(f"model_args.peft_path: {model_args.peft_path}")
    print(f"data_args.train_file: {data_args.train_file}")
    print(f"training_args.do_train: {training_args.do_train}")
    print(f"training_args.do_predict: {training_args.do_predict}")
    
    # 检查是否有其他可能的参数
    print(f"model_args.model_name_or_path: {model_args.model_name_or_path}")
    print(f"training_args.output_dir: {training_args.output_dir}")
    print("==================")
    
    # 手动检查命令行参数
    import sys
    print("=== 命令行参数 ===")
    for i, arg in enumerate(sys.argv):
        if 'peft' in arg.lower() or 'checkpoint' in arg.lower():
            print(f"参数 {i}: {arg}")
    print("==================")

    # ✅ 直接使用配置文件中的路径，不再自动修复或推断
    print("=== 数据路径检查 ===")
    print(f"Train file: {data_args.train_file}")
    print(f"Validation file: {data_args.validation_file}")
    print(f"Test file: {data_args.test_file}")
    print("==================")

    # 确保 train_file 存在
    if not data_args.train_file:
        raise ValueError("❌ 配置文件中未指定 train_file，无法初始化 EHRTokenizer！")

    # 初始化 EHRTokenizer
    # 🔹 如果提供了药物嵌入文件，自动启用药物知识并过滤 ATC4 codes
    drug_emb_file = model_args.drug_embedding_file  # 可能是 None
    ehr_tokenizer = EHRTokenizer(data_args.train_file, drug_embedding_file=drug_emb_file)

    # 组织数据文件路径
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file


    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # 对训练集应用样本平衡策略
    if training_args.do_train and data_args.balance_strategy != "none":
        print(f"应用样本不均衡处理策略: {data_args.balance_strategy}")
        raw_datasets["train"] = apply_balancing_strategy(
            raw_datasets["train"],
            ehr_tokenizer,
            balance_strategy=data_args.balance_strategy
        )

    print("raw_datasets: ", raw_datasets)

    ## Load Model ##
    # 先加载基础模型到CPU，避免CUDA内存不足
    print(f"🔍 检测模型路径: {model_args.model_name_or_path}")
    
    # 🔹 根据路径自动选择模型类型
    if "qwen" in model_args.model_name_or_path.lower():
        print("🤖 使用 QwenForMedRec")
        from llm.qwen import QwenForMedRec
        base_model = QwenForMedRec.from_pretrained(
            model_args.model_name_or_path,
            med_voc_l1=len(ehr_tokenizer.level1_voc.word2idx),
            med_voc_l2=len(ehr_tokenizer.level2_voc.word2idx),
            med_voc_l3=len(ehr_tokenizer.level3_voc.word2idx),
            med_voc_l4=len(ehr_tokenizer.level4_voc.word2idx),
            voc_l1=ehr_tokenizer.level1_voc,
            voc_l2=ehr_tokenizer.level2_voc,
            voc_l3=ehr_tokenizer.level3_voc,
            voc_l4=ehr_tokenizer.level4_voc,
            use_cross_attention=model_args.use_cross_attention,
            cross_threshold=model_args.cross_threshold,
            # 🔬 药物知识融合参数
            use_drug_knowledge=(model_args.drug_embedding_file is not None),
            drug_embedding_file=model_args.drug_embedding_file,
            hierarchical_prediction=model_args.hierarchical_prediction,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # FP16 训练
            low_cpu_mem_usage=False,    # ZeRO-3 要求
        )
    else:
        print("🦙 使用 LlamaForMedRec")
        from llm.llama import LlamaForMedRec
        base_model = LlamaForMedRec.from_pretrained(
            model_args.model_name_or_path,
            med_voc_l1=len(ehr_tokenizer.level1_voc.word2idx),
            med_voc_l2=len(ehr_tokenizer.level2_voc.word2idx),
            med_voc_l3=len(ehr_tokenizer.level3_voc.word2idx),
            med_voc_l4=len(ehr_tokenizer.level4_voc.word2idx),
            voc_l1=ehr_tokenizer.level1_voc,
            voc_l2=ehr_tokenizer.level2_voc,
            voc_l3=ehr_tokenizer.level3_voc,
            voc_l4=ehr_tokenizer.level4_voc,
            use_cross_attention=model_args.use_cross_attention,
            cross_threshold=model_args.cross_threshold,
            # 🔬 药物知识融合参数
            use_drug_knowledge=(model_args.drug_embedding_file is not None),
            drug_embedding_file=model_args.drug_embedding_file,
            hierarchical_prediction=model_args.hierarchical_prediction,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # FP16 训练
            low_cpu_mem_usage=False,    # ZeRO-3 要求
        )
    
    print("✅ 模型加载成功！")
    
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 检查GPU内存状态
    if torch.cuda.is_available():

        print(f"GPU内存状态 - 加载模型前:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  已保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  总容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 启用梯度检查点以节省内存
    base_model.gradient_checkpointing_enable()
    
    # ✅ 使用 DeepSpeed ZeRO-3 + FP16，先不移动到 GPU，让 DeepSpeed 自动管理
    # 保持模型在 CPU，DeepSpeed 会按需加载到 GPU 并转换为 FP16
    base_model = base_model.half()  # FP16 训练（更快，省显存）
    print("✅ 模型保持在 CPU (FP16)，等待 DeepSpeed ZeRO-3 管理显存")
    
    # 清理CPU内存
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 添加频率权重到基础模型
    base_model.freq_weights_l1 = ehr_tokenizer.freq_weights_l1
    base_model.freq_weights_l2 = ehr_tokenizer.freq_weights_l2
    base_model.freq_weights_l3 = ehr_tokenizer.freq_weights_l3
    base_model.freq_weights_l4 = ehr_tokenizer.freq_weights_l4

    print(f"DEBUG: model_args.peft_path = {model_args.peft_path}")
    print(f"DEBUG: model_args.peft_path type = {type(model_args.peft_path)}")
    print(f"DEBUG: model_args.peft_path == '' = {model_args.peft_path == ''}")
    print(f"DEBUG: model_args.peft_path is None = {model_args.peft_path is None}")
    print(f"DEBUG: training_args.resume_from_checkpoint = {training_args.resume_from_checkpoint}")
    
    # 优先使用用户指定的peft_path，如果没有则自动构建
    if model_args.peft_path and model_args.peft_path.strip():
        # 使用用户指定的peft_path
        checkpoint_path = model_args.peft_path
        print(f"🔍 使用用户指定的权重文件路径: {checkpoint_path}")
    else:
        # 从训练参数中获取输出目录和步数
        output_dir = training_args.output_dir  # 例如: "saved/lora-mul"
        max_steps = training_args.max_steps if hasattr(training_args, 'max_steps') else 3000
        
        # 构建权重文件路径
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{max_steps}")
        print(f"🔍 自动构建权重文件路径: {checkpoint_path}")
    
    print(f"🔍 路径是否存在: {os.path.exists(checkpoint_path)}")
    
    if os.path.exists(checkpoint_path):
        print(f"✅ 找到权重文件: {checkpoint_path}")
        print(f"🚀 加载已保存的权重: {checkpoint_path}")
        if training_args.resume_from_checkpoint is not None:
            model = PeftModelForCLS.from_pretrained(base_model, checkpoint_path, is_trainable=True)
        else:
            model = PeftModelForCLS.from_pretrained(base_model, checkpoint_path, is_trainable=False)
        print("✅ 成功加载权重，将进行预测")
        
        # 自动切换到预测模式
        training_args.do_train = False
        training_args.do_predict = True
        print("🔄 自动切换到预测模式")
    else:
        print(f"❌ 权重文件不存在: {checkpoint_path}")
        print("将进行训练...")
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.trainable.split(","),
            lora_dropout=model_args.lora_dropout,
            task_type="SEQ_CLS",
        )
        model = PeftModelForCLS(base_model, peft_config)

    if training_args.do_train:
        for name, param in model.named_parameters():
            if "cls_head" in name:
                param.requires_grad = True
    model.print_trainable_parameters()

    ## Load Tokenizer ##
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    # ✅ 设置pad_token（Qwen使用eos_token，LLaMA使用unk_token）
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # 最后兜底：添加一个新的pad_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"
    
    # ✅ 确保模型config也同步更新pad_token_id
    print(f"✅ Tokenizer设置: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")
    
    # ✅ 同步模型config（重要！防止config.pad_token_id仍为None）
    if hasattr(base_model, 'config'):
        if base_model.config.pad_token_id != tokenizer.pad_token_id:
            print(f"⚠️ 同步模型config: pad_token_id {base_model.config.pad_token_id} -> {tokenizer.pad_token_id}")
            base_model.config.pad_token_id = tokenizer.pad_token_id

    ## Load Dataset ##
    # 数据已经在上面加载过了，这里不需要重复加载

    if training_args.do_train:
        target_dataset = raw_datasets["train"]
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        target_dataset = raw_datasets["validation"]
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        target_dataset = raw_datasets["test"]
        column_names = raw_datasets["test"].column_names

    # 创建数据预处理器
    preprocess_func = llama_train_cls(
        data_args,
        model_args,
        tokenizer,
        ehr_tokenizer,
        hierarchical_prediction=model_args.hierarchical_prediction,
    )
    data_collator = LongestSequenceCollator(tokenizer)

    with training_args.main_process_first(desc="Dataset map pre-processing"):
        target_dataset = target_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col not in ['labels_l1', 'labels_l2', 'labels_l3', 'labels_l4', 'mask_l2', 'mask_l3', 'mask_l4']],
            desc="Running tokenizer on dataset",
            load_from_cache_file=False,  # 禁用缓存避免序列化问题
        )
    target_dataset.set_format("torch")
    print("设置格式后的详细检查:")
    print("数据集特征:", target_dataset.features)

    for i in range(min(3, len(target_dataset))):
        sample = target_dataset[i]
        print(f"\n样本{i}:")
        print(f"  所有keys: {list(sample.keys())}")
        print(f"  各字段类型:")
        for key in sample.keys():
            print(f"    {key}: {type(sample[key])}")


    # 检查预处理结果
    print("预处理后数据集特征:", target_dataset.features)
    if len(target_dataset) > 0:
        sample = target_dataset[0]
        print("样本keys:", list(sample.keys()))


    ## Set Trainer ##
    # 🔥 禁用DeepSpeed保存完整训练状态（节省时间和空间）
    if training_args.do_train and hasattr(training_args, 'deepspeed'):
        # 只在我们的callback中保存PEFT权重，不要让Trainer保存完整状态
        training_args.save_safetensors = False  # 不保存safetensors格式
        print("🔧 已配置：只保存LoRA adapter，不保存完整模型")
    
    trainer = HierarchicalMedRecTrainer(
        model=model,
        args=training_args,
        train_dataset=target_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
        hierarchical_prediction=model_args.hierarchical_prediction,
    )
    
    # 预测前清理GPU缓存和优化内存
    if training_args.do_predict:
        import gc
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        training_args.deepspeed = None
        # 设置内存优化
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 强制减少批处理大小以避免内存不足
        original_batch_size = training_args.per_device_eval_batch_size
        training_args.per_device_eval_batch_size = 1  # 强制设置为1
        print(f"预测时批处理大小: {original_batch_size} -> {training_args.per_device_eval_batch_size} (为避免内存不足)")
        
        # 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 临时减少序列长度以节省内存
        original_max_length = data_args.max_source_length
        data_args.max_source_length = 1024  # 减少到1024
        print(f"临时减少序列长度: {original_max_length} -> {data_args.max_source_length} (为节省内存)")
        
        # 启用梯度检查点以节省内存
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # 预测时禁用梯度计算
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # 清理不必要的缓存
        if hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'model'):
                # 清理Llama模型的缓存
                if hasattr(model.base_model.model, 'config'):
                    # 减少序列长度以减少内存使用
                    if hasattr(model.base_model.model.config, 'max_position_embeddings'):
                        print(f"模型最大位置嵌入: {model.base_model.model.config.max_position_embeddings}")
        
        print("已清理GPU缓存并优化内存设置")

    ## Train Model
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()

    ## Evaluation ##
    results = {}

    if training_args.do_predict:
        # 🎲 确保预测可复现（再次设置随机种子）
        print("🔒 预测模式：重新设置随机种子确保可复现性")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        
        # 强制单进程数据处理
        data_args.preprocessing_num_workers = 1
        print(f"🔒 预测模式：数据预处理使用单进程（num_workers=1）")
        
        # 清空注意力权重缓存
        if hasattr(trainer, 'attention_weights_cache'):
            trainer.attention_weights_cache = []
            print("🧹 已清空注意力权重缓存")
        
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        print(f"📊 测试数据统计:")
        print(f"  - 原始测试样本数: {len(list_test_samples)}")
        print(f"  - 预处理后数据集长度: {len(target_dataset)}")
        print(f"  - 批处理大小: {training_args.per_device_eval_batch_size}")
        print(f"  - GPU数量: {training_args.n_gpu}")
        print(f"  - 分布式训练: {training_args.local_rank != -1}")
        print(f"  - 预期批次数: {len(target_dataset) // training_args.per_device_eval_batch_size + (1 if len(target_dataset) % training_args.per_device_eval_batch_size > 0 else 0)}")
        
        # 检查数据集是否为空
        if len(target_dataset) == 0:
            print("❌ 错误：预处理后数据集为空！")
            return {}
        
        # 检查前几个样本
        print(f"📋 数据集样本检查:")
        for i in range(min(3, len(target_dataset))):
            sample = target_dataset[i]
            print(f"  样本{i}: keys={list(sample.keys())}")
            if 'input_ids' in sample:
                print(f"    input_ids形状: {sample['input_ids'].shape}")
            if 'labels_l1' in sample:
                print(f"    labels_l1形状: {sample['labels_l1'].shape}")

        # 添加内存监控函数
        def monitor_memory():
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                print(f"GPU内存使用: {allocated:.2f}GB 已分配, {reserved:.2f}GB 已保留")
        
        print("开始预测前内存状态:")
        monitor_memory()
        
        start_time = time.time()
        with torch.no_grad():
            # 分批处理以避免内存不足
            print("开始分批预测以避免内存不足...")
            
            # 将数据集分成小批次
            batch_size = 50  # 每次处理50个样本
            total_samples = len(target_dataset)
            all_predictions = []
            
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                batch_dataset = target_dataset.select(range(i, end_idx))
                
                print(f"处理批次 {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}: 样本 {i}-{end_idx-1}")
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 预测当前批次
                batch_results = trainer.predict(
                    batch_dataset,
                    metric_key_prefix="predict",
                )
                
                if batch_results.predictions is not None:
                    all_predictions.append(batch_results.predictions)
                
                # 再次清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                monitor_memory()
            
            # 合并所有预测结果
            if all_predictions:
                if isinstance(all_predictions[0], tuple):
                    # 层级预测结果
                    merged_predictions = tuple(
                        np.concatenate([pred[i] for pred in all_predictions], axis=0)
                        for i in range(len(all_predictions[0]))
                    )
                else:
                    # 单一预测结果
                    merged_predictions = np.concatenate(all_predictions, axis=0)
                
                # 创建预测结果对象
                from transformers.trainer_utils import PredictionOutput
                predict_results = PredictionOutput(
                    predictions=merged_predictions,
                    label_ids=None,
                    metrics={}
                )
            else:
                print("❌ 错误：没有获得任何预测结果")
                return {}
                
        end_time = time.time()
        
        print("预测完成后内存状态:")
        monitor_memory()

        if trainer.is_world_process_zero():
            predictions = predict_results.predictions
            labels = predict_results.label_ids

            print(f"预测完成，耗时: {end_time - start_time:.2f}秒")
            print(f"predict_results类型: {type(predict_results)}")
            print(f"predictions类型: {type(predictions)}")
            print(f"labels类型: {type(labels)}")
            
            # 安全检查
            if predictions is None:
                print("❌ 错误：预测结果为None")
                return {}
            
            # 预测模式下labels可能为None，这是正常的
            if labels is None:
                print("⚠️ 注意：预测模式下标签为None，这是正常的")
                labels = None  # 保持为None，后续处理时会跳过标签相关操作
            
            print(f"预测样本数: {len(predictions)}, 原始样本数: {len(list_test_samples)}")
            
            # 根据hierarchical_prediction处理预测结果
            if model_args.hierarchical_prediction:
                # 层级预测模式：应该返回4元组 (L1, L2, L3, L4)
                if isinstance(predictions, tuple) and len(predictions) == 4:
                    pred_l1, pred_l2, pred_l3, pred_l4 = predictions
                    print(f"✅ 层级预测模式 - 预测结果形状: L1={pred_l1.shape}, L2={pred_l2.shape}, L3={pred_l3.shape}, L4={pred_l4.shape}")
                    
                    # 检查预测值范围
                    print(f"L1预测值范围: min={pred_l1.min():.4f}, max={pred_l1.max():.4f}")
                    print(f"L2预测值范围: min={pred_l2.min():.4f}, max={pred_l2.max():.4f}")
                    print(f"L3预测值范围: min={pred_l3.min():.4f}, max={pred_l3.max():.4f}")
                    print(f"L4预测值范围: min={pred_l4.min():.4f}, max={pred_l4.max():.4f}")
                    
                    # 分层预测分析
                    print("\n=== 分层预测分析 ===")
                    
                    # 分析每层预测结果
                    for level, pred in enumerate([pred_l1, pred_l2, pred_l3, pred_l4], 1):
                        # 确保pred是tensor类型
                        if isinstance(pred, np.ndarray):
                            pred = torch.from_numpy(pred)
                        
                        # 应用sigmoid激活
                        pred_probs = torch.sigmoid(pred)
                        
                        # 计算每层的预测统计（与训练时保持一致）
                        threshold = 0.4# 与训练时的阈值保持一致
                        pred_binary = (pred_probs > threshold).float()
                        num_predicted = pred_binary.sum(dim=1).mean().item()
                        
                        print(f"L{level} 预测统计:")
                        print(f"  - 平均预测数量: {num_predicted:.2f}")
                        print(f"  - 预测概率范围: {pred_probs.min():.4f} - {pred_probs.max():.4f}")
                        
                        # 显示前几个样本的预测结果
                        if level <= 2:  # 只显示L1和L2的详细结果
                            for i in range(min(3, pred_probs.shape[0])):
                                top_indices = torch.topk(pred_probs[i], k=5).indices
                                top_probs = torch.topk(pred_probs[i], k=5).values
                                print(f"  样本{i} Top5预测: {top_indices.cpu().numpy()} (概率: {top_probs.cpu().numpy()})")
                    
                    # 使用L4预测结果作为最终预测
                    final_predictions = pred_l4
                    
                    # 保存分层预测结果用于分析
                    hierarchical_predictions = {
                        'L1': pred_l1,
                        'L2': pred_l2, 
                        'L3': pred_l3,
                        'L4': pred_l4
                    }
                else:
                    # 如果配置了层级预测但返回的不是4元组，可能是错误
                    print(f"⚠️ 警告：配置了层级预测模式，但返回的预测结果不是4元组，而是: {type(predictions)}")
                    if isinstance(predictions, tuple):
                        print(f"   元组长度: {len(predictions)}")
                    # 尝试处理：如果是单一结果，当作L4处理
                    final_predictions = predictions if not isinstance(predictions, tuple) else predictions[-1]
                    print(f"   使用单一预测结果作为L4，形状: {final_predictions.shape}")
                    hierarchical_predictions = None
            else:
                # 非层级预测模式：只返回L4的logits
                if isinstance(predictions, tuple):
                    # 如果返回的是元组，取最后一个（L4）
                    final_predictions = predictions[-1]
                    print(f"⚠️ 非层级模式但返回了元组，使用最后一个元素作为L4预测")
                else:
                    # 直接使用预测结果作为L4
                    final_predictions = predictions
                
                print(f"✅ 非层级预测模式 - 直接预测L4，预测结果形状: {final_predictions.shape}")
                print(f"预测值范围: min={final_predictions.min():.4f}, max={final_predictions.max():.4f}")
                hierarchical_predictions = None

            output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")

            # 获取注意力权重（如果有的话）
            attention_weights_list = trainer.attention_weights_cache if hasattr(trainer, 'attention_weights_cache') else []
            
            # 统计有效的注意力权重数量
            valid_attention_count = sum(1 for attn in attention_weights_list if attn is not None)
            print(f"📊 收集到 {len(attention_weights_list)} 个样本，其中 {valid_attention_count} 个包含注意力权重")
            
            # 如果没有注意力权重，打印诊断信息
            if valid_attention_count == 0 and len(attention_weights_list) > 0:
                print("⚠️ 诊断信息：")
                print(f"   - attention_weights_cache 长度: {len(attention_weights_list)}")
                print(f"   - 第一个元素: {attention_weights_list[0]}")
                print(f"   - 模型是否使用交叉注意力: {model.config.use_cross_attention if hasattr(model.config, 'use_cross_attention') else 'Unknown'}")

            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for idx, p in enumerate(final_predictions):
                    samp = list_test_samples[idx]

                    # 根据evaluate_jsonlines函数的期望格式
                    samp["target"] = p.astype(float).tolist()  # 预测概率向量
                    
                    # 从atc_level_4构建drug_code字段（评估函数需要）
                    if "atc_level_4" in samp:
                        samp["drug_code"] = samp["atc_level_4"]  # 使用L4标签作为drug_code
                        print(f"样本{idx}: 使用atc_level_4作为drug_code: {samp['atc_level_4'][:5]}...")  # 显示前5个
                    else:
                        print(f"警告：样本{idx}缺少atc_level_4字段")
                        samp["drug_code"] = []

                    # 可选调试信息 - 只有在有标签时才添加
                    if labels is not None:
                        if isinstance(labels, tuple) and len(labels) == 4:
                            samp["_true_labels_vector"] = labels[3][idx].astype(float).tolist()  # L4标签
                        else:
                            samp["_true_labels_vector"] = labels[idx].astype(float).tolist()
                    else:
                        # 预测模式下没有真实标签，使用空列表
                        samp["_true_labels_vector"] = []
                    
                    # 添加注意力权重（如果有的话）
                    if idx < len(attention_weights_list) and attention_weights_list[idx] is not None:
                        attn = attention_weights_list[idx]
                        samp["attention_weights"] = {
                            key: value.tolist() for key, value in attn.items()
                        }

                    res = json.dumps(samp, ensure_ascii=False)
                    writer.write(f"{res}\n")

            # 对L4层进行最终评估
            results = evaluate_jsonlines(output_prediction_file, ehr_tokenizer)
            print(f"L4层评估结果: {results}")
            
            # 分层预测效果分析（仅在层级预测模式下）
            if 'hierarchical_predictions' in locals() and hierarchical_predictions is not None:
                analyze_hierarchical_predictions(hierarchical_predictions, list_test_samples, ehr_tokenizer)
                
                # 对每个层级进行独立评估
                evaluate_hierarchical_levels(hierarchical_predictions, list_test_samples, ehr_tokenizer)
            else:
                print("ℹ️ 非层级预测模式，跳过分层预测分析")

    return results

def analyze_hierarchical_predictions(hierarchical_predictions, test_samples, ehr_tokenizer):
    """分析分层预测效果"""
    print("\n=== 分层预测效果分析 ===")
    
    # 分析每层的预测准确性
    for level in ['L1', 'L2', 'L3', 'L4']:
        if level in hierarchical_predictions:
            pred = hierarchical_predictions[level]
            # 确保pred是tensor类型
            if isinstance(pred, np.ndarray):
                pred = torch.from_numpy(pred)
            pred_probs = torch.sigmoid(pred)
            
            print(f"\n{level} 层预测分析:")
            
            # 计算每层的预测统计（与训练时保持一致）
            threshold = 0.4# 与训练时的阈值保持一致
            pred_binary = (pred_probs > threshold).float()
            
            # 分析前几个样本的预测结果
            for i in range(min(5, len(test_samples))):
                sample = test_samples[i]
                true_labels = sample.get(f"atc_level_{level[-1]}", [])
                
                # 获取预测的top-k结果
                top_k = 10
                top_indices = torch.topk(pred_probs[i], k=top_k).indices
                top_probs = torch.topk(pred_probs[i], k=top_k).values
                
                print(f"  样本{i}:")
                print(f"    真实标签: {true_labels}")
                print(f"    Top{top_k}预测: {top_indices.cpu().numpy()}")
                print(f"    预测概率: {top_probs.cpu().numpy()}")
                
                # 计算预测准确性
                if true_labels:
                    # 这里可以添加更详细的准确性分析
                    pass

def evaluate_hierarchical_levels(hierarchical_predictions, test_samples, ehr_tokenizer):
    """对每个层级进行独立评估"""
    print("\n=== 层级独立评估 ===")
    
    from sklearn.metrics import jaccard_score, precision_recall_curve, auc
    
    for level in ['L1', 'L2', 'L3', 'L4']:
        if level in hierarchical_predictions:
            pred = hierarchical_predictions[level]
            # 确保pred是tensor类型
            if isinstance(pred, np.ndarray):
                pred = torch.from_numpy(pred)
            pred_probs = torch.sigmoid(pred).cpu().numpy()
            
            print(f"\n{level} 层独立评估:")
            
            # 获取对应的词汇表
            if level == 'L1':
                vocab = ehr_tokenizer.level1_voc
            elif level == 'L2':
                vocab = ehr_tokenizer.level2_voc
            elif level == 'L3':
                vocab = ehr_tokenizer.level3_voc
            else:  # L4
                vocab = ehr_tokenizer.level4_voc
            
            # 构建真实标签矩阵
            true_labels = []
            for sample in test_samples:
                level_key = f"atc_level_{level[-1]}"
                if level_key in sample:
                    # 将ATC码转换为one-hot向量
                    label_vec = np.zeros(len(vocab.word2idx))
                    for code in sample[level_key]:
                        if code in vocab.word2idx:
                            label_vec[vocab.word2idx[code]] = 1
                    true_labels.append(label_vec)
                else:
                    true_labels.append(np.zeros(len(vocab.word2idx)))
            
            true_labels = np.array(true_labels)
            
            # 计算评估指标（与训练时保持一致）
            threshold = 0.4# 与训练时的阈值保持一致
            pred_binary = (pred_probs > threshold).astype(int)
            
            # Jaccard相似度
            jaccard_scores = []
            for i in range(len(true_labels)):
                if true_labels[i].sum() > 0:  # 只计算有标签的样本
                    jaccard = jaccard_score(true_labels[i], pred_binary[i], average='binary')
                    jaccard_scores.append(jaccard)
            
            avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
            
            # 精确率和召回率
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for i in range(len(true_labels)):
                if true_labels[i].sum() > 0:
                    tp = np.sum((true_labels[i] == 1) & (pred_binary[i] == 1))
                    fp = np.sum((true_labels[i] == 0) & (pred_binary[i] == 1))
                    fn = np.sum((true_labels[i] == 1) & (pred_binary[i] == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
            
            avg_precision = np.mean(precision_scores) if precision_scores else 0
            avg_recall = np.mean(recall_scores) if recall_scores else 0
            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            
            # PRAUC
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(
                    true_labels.flatten(), pred_probs.flatten()
                )
                prauc = auc(recall_curve, precision_curve)
            except:
                prauc = 0
            
            print(f"  Jaccard: {avg_jaccard:.4f}")
            print(f"  Precision: {avg_precision:.4f}")
            print(f"  Recall: {avg_recall:.4f}")
            print(f"  F1: {avg_f1:.4f}")
            print(f"  PRAUC: {prauc:.4f}")
            
            # 预测统计
            avg_pred_count = np.mean(np.sum(pred_binary, axis=1))
            avg_true_count = np.mean(np.sum(true_labels, axis=1))
            print(f"  平均预测数量: {avg_pred_count:.2f}")
            print(f"  平均真实数量: {avg_true_count:.2f}")
            
            # 显示前几个样本的预测结果
            print(f"  前3个样本预测结果:")
            for i in range(min(3, len(test_samples))):
                sample = test_samples[i]
                level_key = f"atc_level_{level[-1]}"
                true_codes = sample.get(level_key, [])
                
                # 获取预测的top-k结果
                top_k = 5
                top_indices = np.argsort(pred_probs[i])[-top_k:][::-1]
                top_probs = pred_probs[i][top_indices]
                
                # 将索引转换为ATC码
                pred_codes = []
                for idx in top_indices:
                    for code, code_idx in vocab.word2idx.items():
                        if code_idx == idx:
                            pred_codes.append(code)
                            break
                
                print(f"    样本{i}:")
                print(f"      真实: {true_codes}")
                print(f"      预测: {pred_codes}")
                print(f"      概率: {top_probs}")


if __name__ == "__main__":
    train()