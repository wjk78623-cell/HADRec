# here put the import lib
import numpy as np
import pandas as pd
import pickle
import copy
import os
import random
import dill

import torch
from torch.utils.data import Dataset


class Voc:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_freq = {}  # 记录每个 code 的出现频率

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
            self.word_freq[word] = self.word_freq.get(word, 0) + 1


class EHRTokenizer:
    def __init__(self, voc_dir, drug_embedding_file=None):
        self.voc_dir = voc_dir
        self.drug_embedding_file = drug_embedding_file
        self.level1_voc = Voc()
        self.level2_voc = Voc()
        self.level3_voc = Voc()
        self.level4_voc = Voc()  # 即 drug_code
        self.med_voc = Voc()  # 保持兼容性
        
        # 从训练数据构建词汇表
        self.build_vocs_from_data()
        
        # 🔹 如果提供了药物嵌入文件，过滤掉不在映射中的 ATC4 code
        if drug_embedding_file is not None:
            self._filter_atc4_by_drug_mapping()
        
        # 计算频率权重
        self.freq_weights_l1 = self._compute_inverse_frequency_weights(self.level1_voc)
        self.freq_weights_l2 = self._compute_inverse_frequency_weights(self.level2_voc)
        self.freq_weights_l3 = self._compute_inverse_frequency_weights(self.level3_voc)
        self.freq_weights_l4 = self._compute_inverse_frequency_weights(self.level4_voc)

        # 构建层级映射关系
        self.build_hierarchy_mapping()
    
    def build_vocs(self):
        """兼容性方法：调用build_vocs_from_data"""
        return self.build_vocs_from_data()

    def build_vocs_from_data(self):
        """从训练数据构建词汇表"""
        import json
        import os
        
        # 直接使用传入的voc_dir作为训练文件路径
        train_file = self.voc_dir
        
        # 修复路径问题：清理路径中的多余斜杠
        if train_file:
            # 只修复双斜杠问题，保留ccu_mul目录
            train_file = train_file.replace("//", "/")
        
        if train_file:
            print(f"从 {train_file} 构建词汇表")
            with open(train_file, "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    
                    # 构建各层级词汇表
                    if "atc_level_1" in sample:
                        self.level1_voc.add_sentence(sample["atc_level_1"])
                    if "atc_level_2" in sample:
                        self.level2_voc.add_sentence(sample["atc_level_2"])
                    if "atc_level_3" in sample:
                        self.level3_voc.add_sentence(sample["atc_level_3"])
                    if "atc_level_4" in sample:
                        self.level4_voc.add_sentence(sample["atc_level_4"])
                        # 同时添加到med_voc保持兼容性
                        self.med_voc.add_sentence(sample["atc_level_4"])
                    elif "drug_code" in sample:
                        # 如果只有drug_code，将其作为L4处理
                        self.level4_voc.add_sentence(sample["drug_code"])
                        self.med_voc.add_sentence(sample["drug_code"])
                    elif "target" in sample:
                        # 如果只有target，尝试解析药品名称
                        target_text = sample["target"]
                        if isinstance(target_text, str):
                            drug_names = [name.strip() for name in target_text.split(",") if name.strip()]
                            self.level4_voc.add_sentence(drug_names)
                            self.med_voc.add_sentence(drug_names)
        else:
            print("警告：未找到训练文件，使用空词汇表")
            # 创建基本的空词汇表
            self.level1_voc.add_sentence(["A", "B", "C", "D", "H", "J", "N", "P", "R", "S", "V"])
            self.level2_voc.add_sentence(["A11", "B03", "N02", "N03", "N05", "N06"])
            self.level3_voc.add_sentence(["A11D", "B03B", "N02B", "N03A", "N05B", "N06A"])
            self.level4_voc.add_sentence(["A11DA", "B03BB", "N02BA", "N02BE", "N03AE", "N05BA", "N06AA", "N06AB", "N06AX"])
            self.med_voc.add_sentence(["A11DA", "B03BB", "N02BA", "N02BE", "N03AE", "N05BA", "N06AA", "N06AB", "N06AX"])

    def _filter_atc4_by_drug_mapping(self):
        """
        过滤掉不在药物嵌入映射文件中的 ATC4 code
        """
        import os
        import torch
        
        if not os.path.exists(self.drug_embedding_file):
            print(f"⚠️ 药物嵌入文件不存在: {self.drug_embedding_file}，跳过过滤")
            return
        
        print(f"🔍 检查 ATC4 codes 与药物嵌入映射的匹配情况...")
        
        # 加载药物嵌入文件
        drug_data = torch.load(self.drug_embedding_file, map_location='cpu')
        valid_atc4_set = set(drug_data.get("atc4_to_idx", {}).keys())
        
        # 统计原始数量
        original_count = len(self.level4_voc.idx2word)
        
        # 找出不在映射中的 ATC4 codes
        removed_codes = []
        new_word2idx = {}
        new_idx2word = []
        new_word_freq = {}
        
        for old_idx, atc4_code in enumerate(self.level4_voc.idx2word):
            if atc4_code in valid_atc4_set:
                # 保留这个 code
                new_idx = len(new_idx2word)
                new_idx2word.append(atc4_code)
                new_word2idx[atc4_code] = new_idx
                new_word_freq[atc4_code] = self.level4_voc.word_freq.get(atc4_code, 0)
            else:
                # 移除这个 code
                removed_codes.append(atc4_code)
        
        # 更新词汇表
        self.level4_voc.idx2word = new_idx2word
        self.level4_voc.word2idx = new_word2idx
        self.level4_voc.word_freq = new_word_freq
        
        # 同步更新 med_voc
        self.med_voc.idx2word = new_idx2word
        self.med_voc.word2idx = new_word2idx
        self.med_voc.word_freq = new_word_freq
        
        # 输出统计信息
        filtered_count = len(new_idx2word)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            print(f"⚠️ 过滤掉 {removed_count} 个不在药物映射中的 ATC4 codes:")
            print(f"   原始数量: {original_count}")
            print(f"   保留数量: {filtered_count}")
            print(f"   移除的codes: {', '.join(removed_codes[:10])}{'...' if len(removed_codes) > 10 else ''}")
        else:
            print(f"✅ 所有 {original_count} 个 ATC4 codes 都在药物映射中")
    
    def build_hierarchy_mapping(self):
        """构建ATC码的层级映射关系"""
        from collections import defaultdict
        import json
        import os
        
        self.l1_to_l2 = defaultdict(list)
        self.l2_to_l3 = defaultdict(list)
        self.l3_to_l4 = defaultdict(list)

        # 从训练数据中提取层级关系
        train_file = self.voc_dir
        
        # 修复路径问题：清理路径中的多余斜杠
        if train_file:
            # 只修复双斜杠问题，保留ccu_mul目录
            train_file = train_file.replace("//", "/")
                
        if train_file:
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

    def _compute_inverse_frequency_weights(self, voc):
        """计算逆频率权重"""
        freqs = [voc.word_freq.get(w, 1) for w in voc.idx2word]
        weights = 1.0 / np.log(np.array(freqs) + 1)
        return torch.tensor(weights, dtype=torch.float32)

    def get_children_l1_to_l2(self, l1_id):
        """获取L1对应的所有L2子节点"""
        l1_code = self.level1_voc.idx2word[l1_id]
        children_codes = self.l1_to_l2.get(l1_code, [])
        return [self.level2_voc.word2idx[code] for code in children_codes if code in self.level2_voc.word2idx]

    def get_children_l2_to_l3(self, l2_id):
        """获取L2对应的所有L3子节点"""
        l2_code = self.level2_voc.idx2word[l2_id]
        children_codes = self.l2_to_l3.get(l2_code, [])
        return [self.level3_voc.word2idx[code] for code in children_codes if code in self.level3_voc.word2idx]

    def get_children_l3_to_l4(self, l3_id):
        """获取L3对应的所有L4子节点"""
        l3_code = self.level3_voc.idx2word[l3_id]
        children_codes = self.l3_to_l4.get(l3_code, [])
        return [self.level4_voc.word2idx[code] for code in children_codes if code in self.level4_voc.word2idx]

    def convert_med_tokens_to_ids(self, med_tokens):
        """兼容性方法：将药品token转换为ID"""
        ids = []
        for token in med_tokens:
            if token in self.level4_voc.word2idx:
                ids.append(self.level4_voc.word2idx[token])
            else:
                ids.append(-1)  # 未知token
        return ids

    def convert_tokens_to_ids(self, tokens):
        """兼容性方法：将token转换为ID"""
        ids = []
        for token in tokens:
            if token in self.level4_voc.word2idx:
                ids.append(self.level4_voc.word2idx[token])
            elif token == '[PAD]':
                ids.append(0)  # 假设0是PAD token
            else:
                ids.append(-1)  # 未知token
        return ids


class EHRDataset(Dataset):
    '''The dataset for medication recommendation'''

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len  # the maximum length of a diagnosis/procedure record

        self.sample_counter = 0
        self.records = data_pd

        self.var_name = []

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError


####################################
'''Finetune Dataset'''


####################################

class FinetuneEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):

        super().__init__(data_pd, tokenizer, max_seq_len)
        self.max_seq = 10
        self.var_name = ["diag_seq", "proc_seq", "med_seq", "seq_mask", "labels"]

    def __len__(self):

        return len(self.records)

    def __getitem__(self, item):

        # one admission: [diagnosis, procedure, medication]
        adm = copy.deepcopy(self.records[item])

        med_seq = [meta_adm[2] for meta_adm in adm]
        diag_seq = [meta_adm[0] for meta_adm in adm]
        proc_seq = [meta_adm[1] for meta_adm in adm]

        # get the medcation recommendation label -- multi-hot vector
        label_index = self.tokenizer.convert_med_tokens_to_ids(med_seq[-1])
        label = np.zeros(len(self.tokenizer.med_voc.word2idx))
        for index in label_index:
            label[index] = 1

        # get the seq len
        # pad the sequence to longest med / diag / proc sequences
        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        # convert raw tokens to unified ids
        med_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in med_seq]
        diag_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in diag_seq]
        proc_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in proc_seq]

        # pad the sequence to max possible records
        pad_seq = ["[PAD]" for _ in range(self.seq_len)]
        pad_seq = self.tokenizer.convert_tokens_to_ids(pad_seq)

        def fill_to_max_seq(l, seq):
            pad_num = 0
            while len(l) < seq:
                l.append(pad_seq)
                pad_num += 1
            if len(l) > seq:
                l = l[:seq]
            return l, pad_num

        med_seq = med_seq[:-1]  # remove the current medication set, which is label
        med_seq, _ = fill_to_max_seq(med_seq, self.max_seq)
        diag_seq, pad_num = fill_to_max_seq(diag_seq, self.max_seq)
        proc_seq, _ = fill_to_max_seq(proc_seq, self.max_seq)

        # get mask
        mask = np.ones(self.max_seq)
        if pad_num != 0:
            mask[-pad_num:] = 0

        return np.array(diag_seq, dtype=int), np.array(proc_seq, dtype=int), \
            np.array(med_seq, dtype=int), mask.astype(int), label.astype(float)


####################################
'''MedRec Dataset'''


####################################

class MedRecEHRDataset(EHRDataset):

    def __init__(self, data_pd, tokenizer: EHRTokenizer, profile_tokenizer, args):

        super().__init__(data_pd, tokenizer, args.max_seq_length)

        if args.filter:
            self._filter_data()

        self.max_seq = args.max_record_num
        self.profile_tokenizer = profile_tokenizer
        self.var_name = ["diag_seq", "proc_seq", "med_seq", "seq_mask", "labels", "multi_label", "profile"]

    def __len__(self):

        return len(self.records)

    def __getitem__(self, item):

        # one admission: [diagnosis, procedure, medication]
        adm = copy.deepcopy(self.records[item])

        med_seq = adm["records"]["medication"]
        diag_seq = adm["records"]["diagnosis"]
        proc_seq = adm["records"]["procedure"]

        # encode profile, get a vector to organize all feature orderly
        profile = []
        for k, v in adm["profile"].items():
            profile.append(self.profile_tokenizer["word2idx"][k][v])

        # get the medcation recommendation label -- multi-hot vector
        label_index = self.tokenizer.convert_med_tokens_to_ids(med_seq[-1])
        label = np.zeros(len(self.tokenizer.med_voc.word2idx))
        multi_label = np.full(len(self.tokenizer.med_voc.word2idx), -1)
        for i, index in enumerate(label_index):
            label[index] = 1
            multi_label[i] = index

        # get the seq len
        # pad the sequence to longest med / diag / proc sequences
        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        # convert raw tokens to unified ids
        med_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in med_seq]
        diag_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in diag_seq]
        proc_seq = [self.tokenizer.convert_tokens_to_ids(fill_to_max(meta_seq, self.seq_len)) for meta_seq in proc_seq]

        # pad the sequence to max possible records
        pad_seq = ["[PAD]" for _ in range(self.seq_len)]
        pad_seq = self.tokenizer.convert_tokens_to_ids(pad_seq)

        def fill_to_max_seq(l, seq):
            pad_num = 0
            while len(l) < seq:
                l.append(pad_seq)
                pad_num += 1
            if len(l) > seq:
                l = l[:seq]
            return l, pad_num

        med_seq = med_seq[:-1]  # remove the current medication set, which is label
        med_seq, _ = fill_to_max_seq(med_seq, self.max_seq)
        diag_seq, pad_num = fill_to_max_seq(diag_seq, self.max_seq)
        proc_seq, _ = fill_to_max_seq(proc_seq, self.max_seq)

        # get mask
        mask = np.ones(self.max_seq)
        if pad_num != 0:
            mask[-pad_num:] = 0

        return np.array(diag_seq, dtype=int), np.array(proc_seq, dtype=int), \
            np.array(med_seq, dtype=int), mask.astype(int), label.astype(float), \
            multi_label.astype(int), np.array(profile, dtype=int)

    def _filter_data(self):

        new_records = []

        for record in self.records:
            if len(record["records"]["medication"]) > 1:
                new_records.append(record)

        self.records = new_records
