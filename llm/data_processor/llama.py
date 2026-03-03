# here put the import lib
import numpy as np
import json
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class llama_train(object):

    def __init__(self, data_args, model_args, tokenizer) -> None:

        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.history_column = None
        self.tokenizer = tokenizer

    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                if len(b_ids) > self.data_args.max_target_length - 2:
                    b_ids = b_ids[: self.data_args.max_target_length - 2]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
                labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

                pad_len = max_seq_length - len(input_ids)

                if self.data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs


class llama_eval(object):

    def __init__(self, data_args, model_args, tokenizer) -> None:

        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.history_column = None
        self.tokenizer = tokenizer

    def __call__(self, examples):

        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        for i in range(len(examples[self.prompt_column])):
            if not examples[self.response_column][i]:
                targets.append("filled in !")
            else:
                targets.append(examples[self.response_column][i])

            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                if self.history_column is None or len(examples[self.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

        inputs = [inp for inp in inputs]
        model_inputs = self.tokenizer(inputs,
                                      max_length=self.data_args.max_source_length,
                                      truncation=True,
                                      padding=True)
        labels = self.tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


class llama_train_cls(object):
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer, hierarchical_prediction=True):
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer
        self.hierarchical_prediction = hierarchical_prediction
        
        # 构建层级映射关系
        self.build_hierarchy_mapping()

    def build_hierarchy_mapping(self):
        """构建ATC码的层级映射关系"""
        from collections import defaultdict
        import json
        import os
        
        self.l1_to_l2 = defaultdict(list)
        self.l2_to_l3 = defaultdict(list)
        self.l3_to_l4 = defaultdict(list)
        
        # 从训练数据中提取层级关系
        train_file = self.data_args.train_file
        if train_file and os.path.exists(train_file):
            print(f"从训练文件构建层级映射: {train_file}")
            with open(train_file, "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    l1_codes = sample.get("atc_level_1", [])
                    l2_codes = sample.get("atc_level_2", [])
                    l3_codes = sample.get("atc_level_3", [])
                    l4_codes = sample.get("atc_level_4", [])
                    
                    # 构建映射关系（基于ATC码的前缀关系）
                    for l2 in l2_codes:
                        l1_prefix = l2[:1]  # A11 -> A
                        if l1_prefix in l1_codes:
                            if l2 not in self.l1_to_l2[l1_prefix]:
                                self.l1_to_l2[l1_prefix].append(l2)
                    
                    for l3 in l3_codes:
                        l2_prefix = l3[:3]  # A11D -> A11
                        if l2_prefix in l2_codes:
                            if l3 not in self.l2_to_l3[l2_prefix]:
                                self.l2_to_l3[l2_prefix].append(l3)
                    
                    for l4 in l4_codes:
                        l3_prefix = l4[:4]  # A11DA -> A11D
                        if l3_prefix in l3_codes:
                            if l4 not in self.l3_to_l4[l3_prefix]:
                                self.l3_to_l4[l3_prefix].append(l4)

    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels_l4": [],
        }
        if self.hierarchical_prediction:
            model_inputs.update({
                "labels_l1": [], "labels_l2": [], "labels_l3": [],
                "mask_l2": [], "mask_l3": [], "mask_l4": []  # 用于下层预测的掩码
            })

        for i in range(len(examples[self.prompt_column])):
            query = examples[self.prompt_column][i]

            # Tokenize输入文本
            a_ids = self.tokenizer.encode(text=query, add_special_tokens=False)
            if len(a_ids) > max_seq_length - 1:
                a_ids = a_ids[:max_seq_length - 1]

            input_ids = a_ids + [self.tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)

            # Padding
            pad_len = max_seq_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            # 获取层级标签
            l1_labels = examples["atc_level_1"][i]
            l2_labels = examples["atc_level_2"][i]
            l3_labels = examples["atc_level_3"][i]
            l4_labels = examples["atc_level_4"][i]

            # 转为多标签向量
            def to_multi_hot(labels, voc):
                vec = np.zeros(len(voc.word2idx))
                for label in labels:
                    if label in voc.word2idx:
                        vec[voc.word2idx[label]] = 1
                return vec

            labels_l1 = to_multi_hot(l1_labels, self.ehr_tokenizer.level1_voc)
            labels_l2 = to_multi_hot(l2_labels, self.ehr_tokenizer.level2_voc)
            labels_l3 = to_multi_hot(l3_labels, self.ehr_tokenizer.level3_voc)
            labels_l4 = to_multi_hot(l4_labels, self.ehr_tokenizer.level4_voc)

            # 构建层级掩码（用于推理时缩小搜索空间）
            mask_l2 = np.ones(len(self.ehr_tokenizer.level2_voc.word2idx))  # 预测时使用全1
            mask_l3 = np.ones(len(self.ehr_tokenizer.level3_voc.word2idx))  # 预测时使用全1
            mask_l4 = np.ones(len(self.ehr_tokenizer.level4_voc.word2idx))  # 预测时使用全1

            # 在训练时，基于真实标签构建掩码（用于层级一致性）
            # 在预测时，使用全1掩码，允许所有可能的预测
            if hasattr(self, 'is_training') and self.is_training:
                # 训练时使用真实标签构建掩码
                for l1_label in l1_labels:
                    if l1_label in self.ehr_tokenizer.level1_voc.word2idx:
                        l1_id = self.ehr_tokenizer.level1_voc.word2idx[l1_label]
                        children = self.get_children_l1_to_l2(l1_id)
                        for child in children:
                            mask_l2[child] = 1

                for l2_label in l2_labels:
                    if l2_label in self.ehr_tokenizer.level2_voc.word2idx:
                        l2_id = self.ehr_tokenizer.level2_voc.word2idx[l2_label]
                        children = self.get_children_l2_to_l3(l2_id)
                        for child in children:
                            mask_l3[child] = 1

                for l3_label in l3_labels:
                    if l3_label in self.ehr_tokenizer.level3_voc.word2idx:
                        l3_id = self.ehr_tokenizer.level3_voc.word2idx[l3_label]
                        children = self.get_children_l3_to_l4(l3_id)
                        for child in children:
                            mask_l4[child] = 1

            # 移除药物残差信息 - 不再需要

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            if self.hierarchical_prediction:
                model_inputs["labels_l1"].append(labels_l1)
                model_inputs["labels_l2"].append(labels_l2)
                model_inputs["labels_l3"].append(labels_l3)
                model_inputs["mask_l2"].append(mask_l2)
                model_inputs["mask_l3"].append(mask_l3)
                model_inputs["mask_l4"].append(mask_l4)
            model_inputs["labels_l4"].append(labels_l4)

        return model_inputs

    def get_children_l1_to_l2(self, l1_id):
        """获取L1对应的所有L2子节点"""
        l1_code = self.ehr_tokenizer.level1_voc.idx2word[l1_id]
        children_codes = self.l1_to_l2.get(l1_code, [])
        return [self.ehr_tokenizer.level2_voc.word2idx[code] for code in children_codes if code in self.ehr_tokenizer.level2_voc.word2idx]

    def get_children_l2_to_l3(self, l2_id):
        """获取L2对应的所有L3子节点"""
        l2_code = self.ehr_tokenizer.level2_voc.idx2word[l2_id]
        children_codes = self.l2_to_l3.get(l2_code, [])
        return [self.ehr_tokenizer.level3_voc.word2idx[code] for code in children_codes if code in self.ehr_tokenizer.level3_voc.word2idx]

    def get_children_l3_to_l4(self, l3_id):
        """获取L3对应的所有L4子节点"""
        l3_code = self.ehr_tokenizer.level3_voc.idx2word[l3_id]
        children_codes = self.l3_to_l4.get(l3_code, [])
        return [self.ehr_tokenizer.level4_voc.word2idx[code] for code in children_codes if code in self.ehr_tokenizer.level4_voc.word2idx]


def apply_balancing_strategy(dataset, ehr_tokenizer, balance_strategy="combined"):
    """
    应用样本不均衡处理策略
    Args:
        dataset: 原始数据集
        ehr_tokenizer: EHR分词器
        balance_strategy: 平衡策略 ("combined", "oversample", "undersample")
    """
    print(f"应用样本平衡策略: {balance_strategy}")
    
    # 统计每个层级的频率
    l1_freq = defaultdict(int)
    l2_freq = defaultdict(int)
    l3_freq = defaultdict(int)
    l4_freq = defaultdict(int)
    
    # 收集所有样本
    samples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        samples.append(sample)
        
        # 统计频率
        for code in sample.get("atc_level_1", []):
            l1_freq[code] += 1
        for code in sample.get("atc_level_2", []):
            l2_freq[code] += 1
        for code in sample.get("atc_level_3", []):
            l3_freq[code] += 1
        for code in sample.get("atc_level_4", []):
            l4_freq[code] += 1
    
    # 计算频率阈值 - 使用更合理的阈值
    l1_freqs = list(l1_freq.values())
    l2_freqs = list(l2_freq.values())
    l3_freqs = list(l3_freq.values())
    l4_freqs = list(l4_freq.values())
    
    # 使用更低的阈值，确保有足够的低频样本
    l1_threshold = np.percentile(l1_freqs, 20) if l1_freqs else 1  # 20%分位数
    l2_threshold = np.percentile(l2_freqs, 20) if l2_freqs else 1
    l3_threshold = np.percentile(l3_freqs, 20) if l3_freqs else 1
    l4_threshold = np.percentile(l4_freqs, 20) if l4_freqs else 1
    
    print(f"频率阈值 - L1: {l1_threshold}, L2: {l2_threshold}, L3: {l3_threshold}, L4: {l4_threshold}")
    
    # 分类样本
    high_freq_samples = []
    low_freq_samples = []
    
    for sample in samples:
        l1_codes = sample.get("atc_level_1", [])
        l4_codes = sample.get("atc_level_4", [])
        
        # 判断是否为高频样本（基于L1和L4）
        is_high_freq = any(l1_freq.get(code, 0) >= l1_threshold for code in l1_codes) or \
                      any(l4_freq.get(code, 0) >= l4_threshold for code in l4_codes)
        
        if is_high_freq:
            high_freq_samples.append(sample)
        else:
            low_freq_samples.append(sample)
    
    print(f"高频样本数: {len(high_freq_samples)}, 低频样本数: {len(low_freq_samples)}")
    
    balanced_samples = []
    
    if balance_strategy == "combined":
        # 对高频样本进行硬负采样
        if len(high_freq_samples) > len(low_freq_samples) * 2:
            # 随机采样高频样本
            target_high_freq = len(low_freq_samples) * 2
            rng = np.random.RandomState(42)
            high_freq_samples = rng.choice(high_freq_samples, target_high_freq, replace=False).tolist()
        
        # 对低频样本进行过采样
        if len(low_freq_samples) > 0:
            # 使用SMOTE进行过采样
            try:
                # 准备特征（这里简化处理，使用文本长度作为特征）
                X = np.array([[len(sample.get("input", ""))] for sample in low_freq_samples])
                y = np.array([0] * len(low_freq_samples))  # 所有低频样本标记为同一类
                
                # 过采样到目标数量
                target_size = max(len(high_freq_samples), len(low_freq_samples) * 2)
                if target_size > len(low_freq_samples):
                    smote = SMOTE(random_state=42, k_neighbors=min(3, len(low_freq_samples)-1))
                    X_resampled, _ = smote.fit_resample(X, y)
                    
                    # 根据过采样结果复制样本
                    oversampled_samples = []
                    for i in range(len(X_resampled)):
                        if i < len(low_freq_samples):
                            oversampled_samples.append(low_freq_samples[i])
                        else:
                            # 随机选择一个低频样本进行复制
                            idx = i % len(low_freq_samples)
                            oversampled_samples.append(low_freq_samples[idx])
                    
                    low_freq_samples = oversampled_samples
            except Exception as e:
                print(f"SMOTE过采样失败，使用简单复制: {e}")
                # 简单复制低频样本
                while len(low_freq_samples) < len(high_freq_samples):
                    low_freq_samples.extend(low_freq_samples[:min(len(low_freq_samples), len(high_freq_samples) - len(low_freq_samples))])
    
    elif balance_strategy == "oversample":
        # 只对低频样本进行过采样
        if len(low_freq_samples) > 0:
            target_size = len(high_freq_samples) * 2
            while len(low_freq_samples) < target_size:
                low_freq_samples.extend(low_freq_samples[:min(len(low_freq_samples), target_size - len(low_freq_samples))])
    
    elif balance_strategy == "undersample":
        # 只对高频样本进行欠采样
        if len(high_freq_samples) > len(low_freq_samples) * 2:
            target_size = len(low_freq_samples) * 2
            rng = np.random.RandomState(42)
            high_freq_samples = rng.choice(high_freq_samples, target_size, replace=False).tolist()
    
    balanced_samples = high_freq_samples + low_freq_samples
    
    print(f"平衡后样本数: {len(balanced_samples)}")
    
    # 安全检查：如果平衡后样本数为0，返回原始数据集
    if len(balanced_samples) == 0:
        print("警告：平衡后样本数为0，返回原始数据集")
        return dataset
    
    # 创建新的数据集
    from datasets import Dataset
    return Dataset.from_list(balanced_samples)


class llama_eval_cls(object):

    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer) -> None:

        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = getattr(data_args, 'response_column', 'atc_level_4')
        self.history_column = None
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer

    def __call__(self, examples):

        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        for i in range(len(examples[self.prompt_column])):
            # 处理药物名称字符串
            if self.response_column in examples:
                answer = examples[self.response_column][i]
                if isinstance(answer, str):
                    med_names = [med.strip() for med in answer.split(',') if med.strip()]
                    label_index = self.ehr_tokenizer.convert_med_tokens_to_ids(med_names)
                elif isinstance(answer, list):
                    label_index = self.ehr_tokenizer.convert_med_tokens_to_ids(answer)
                else:
                    label_index = []
            else:
                # 如果没有response_column，创建空的标签
                label_index = []

            med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)
            labels = np.zeros((med_voc_size))

            for idx in label_index:
                if idx != -1:
                    labels[idx] = 1

            targets.append(labels)

            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                if self.history_column is None or len(examples[self.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

        inputs = [inp for inp in inputs]
        model_inputs = self.tokenizer(inputs,
                                      max_length=self.data_args.max_source_length,
                                      truncation=True,
                                      padding=True)
        model_inputs["labels"] = targets

        return model_inputs


class llama_dpo_cls(object):

    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer) -> None:

        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.positive_column = "positive"
        self.negative_column = "negative"
        self.history_column = None
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer

    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "prompt_ids": [],
            "chosen": [],
            "rejected_ids": [],
        }

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.positive_column][i]:
                query = examples[self.prompt_column][i]
                positive, negative = examples[self.positive_column][i], examples[self.negative_column][i]

                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)

                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids)

                context_length = len(a_ids)
                input_ids = a_ids + [self.tokenizer.eos_token_id]

                med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)
                positive_labels, negative_labels = np.zeros((med_voc_size)), np.zeros((med_voc_size))
                positive_labels[positive] = 1
                negative_labels[negative] = 1

                model_inputs["prompt_ids"].append(input_ids)
                model_inputs["chosen_ids"].append(positive_labels)
                model_inputs["rejected_ids"].append(negative_labels)

        return model_inputs