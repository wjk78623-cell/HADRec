# here put the import lib
from dataclasses import dataclass
from typing import Any, List, Dict, Sequence, Tuple, Optional
import torch
import transformers
from transformers import DataCollatorForSeq2Seq
import numpy as np

IGNORE_INDEX = -100
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class BalancedDataset:
    def __init__(self, dataset, ehr_tokenizer):
        self.dataset = dataset
        self.ehr_tokenizer = ehr_tokenizer
        
    def apply_balancing(self):
        """应用过采样和欠采样平衡策略"""
        # 分析L4层级频率
        l4_freq = self.ehr_tokenizer.level4_voc.word_freq
        frequencies = list(l4_freq.values())
        
        # 定义阈值
        high_freq_threshold = np.percentile(frequencies, 75)  # 高频药品
        low_freq_threshold = np.percentile(frequencies, 25)   # 低频药品
        
        # 分离高频和低频样本
        high_freq_samples = []
        low_freq_samples = []
        medium_freq_samples = []
        
        for i, sample in enumerate(self.dataset):
            l4_labels = sample["atc_level_4"]
            max_freq = max([l4_freq.get(label, 0) for label in l4_labels])
            
            if max_freq > high_freq_threshold:
                high_freq_samples.append(i)
            elif max_freq < low_freq_threshold:
                low_freq_samples.append(i)
            else:
                medium_freq_samples.append(i)
        
        # 对高频药品进行硬负采样（减少重复）
        if len(high_freq_samples) > len(medium_freq_samples) * 2:
            # 随机丢弃一部分高频样本
            keep_ratio = len(medium_freq_samples) * 2 / len(high_freq_samples)
            high_freq_samples = random.sample(high_freq_samples, 
                                            int(len(high_freq_samples) * keep_ratio))
        
        # 对低频药品应用SMOTE过采样
        if len(low_freq_samples) < len(medium_freq_samples) // 2:
            # 这里可以添加SMOTE过采样逻辑
            # 由于药品数据的特殊性，可能需要定制的过采样方法
            pass
            
        balanced_indices = high_freq_samples + medium_freq_samples + low_freq_samples
        return self.dataset.select(balanced_indices)

@dataclass
class LongestSequenceCollator(object):
    """Collate examples for supervised fine-tuning, supporting drug_residuals."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if len(instances) == 0:
            return {}

        # ✅ 获取padding值（Qwen可能没有pad_token_id，使用eos_token_id代替）
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0  # 最后的兜底方案

        # 提取所有字段
        all_keys = set()
        for instance in instances:
            all_keys.update(instance.keys())

        result = {}
        for key in all_keys:
            if key == "input_ids":
                # padding input_ids
                input_ids = [torch.tensor(instance[key]) for instance in instances]
                result[key] = torch.nn.utils.rnn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=pad_token_id,
                )
            elif key == "labels":
                # padding labels
                labels = [torch.tensor(instance[key]) for instance in instances]
                result[key] = torch.nn.utils.rnn.pad_sequence(
                    labels,
                    batch_first=True,
                    padding_value=IGNORE_INDEX,
                )
            elif key == "drug_residuals":
                # 处理 drug_residuals：转为 float32 并堆叠
                tensor_list = []
                for instance in instances:
                    data = instance[key]
                    if isinstance(data, torch.Tensor):
                        tensor_list.append(data)
                    else:
                        tensor_list.append(torch.tensor(data, dtype=torch.float32))
                # 不做 padding，直接 stack
                result[key] = torch.stack(tensor_list)  # shape: [B, num_drugs]
            else:
                # 其他字段（如 attention_mask）
                tensor_list = []
                for instance in instances:
                    data = instance[key]
                    if isinstance(data, torch.Tensor):
                        tensor_list.append(data)
                    else:
                        tensor_list.append(torch.tensor(data))
                result[key] = torch.stack(tensor_list)

        return result


class DPODataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        concatenated_features = []
        label_positions = []
        for key in ("chosen_ids", "rejected_ids"):
            for feature in features:
                prompt_len, answer_len = len(feature["prompt_ids"]), len(feature[key])
                concatenated_features.append({
                    "input_ids": feature["prompt_ids"] + feature[key],
                    "attention_mask": [1] * (prompt_len + answer_len)
                })
                label_positions.append((prompt_len, answer_len))

        batch = self.tokenizer.pad(
            concatenated_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        return batch