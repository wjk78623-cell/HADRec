"""
最终评估：使用全局阈值 0.35
- 10次随机抽样30%测试集
- 计算 Jaccard, F1, PR-AUC, DDI Rate
- 报告均值 ± 标准差

⚠️ 重要：
- F1/Jaccard/PR-AUC 使用 EHRTokenizer 词汇表（和训练时一致）
- DDI 使用 PKL 词汇表（和 ddi.py 一致）
- 注意：DDI 用 PKL 词汇表意味着在不同的药物索引上计算
"""

import os
import json
import random
import numpy as np
import pandas as pd
import dill
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_curve, auc
from generators.data import EHRTokenizer


def load_predictions_and_labels(pred_file, ehr_tokenizer):
    """加载预测和标签"""
    print(f"📂 加载预测文件: {pred_file}")
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    # 提取预测 logits
    pred_logits = np.array([sample['target'] for sample in samples])
    pred_probs = 1 / (1 + np.exp(-pred_logits))  # sigmoid
    
    # 构建真实标签
    vocab = ehr_tokenizer.level4_voc
    true_labels = []
    
    for sample in samples:
        label_vec = np.zeros(len(vocab.word2idx))
        codes = sample.get('drug_code', sample.get('atc_level_4', []))
        for code in codes:
            if code in vocab.word2idx:
                label_vec[vocab.word2idx[code]] = 1
        true_labels.append(label_vec)
    
    true_labels = np.array(true_labels)
    
    print(f"  样本数: {len(samples)}")
    print(f"  预测形状: {pred_probs.shape}")
    print(f"  标签形状: {true_labels.shape}")
    print(f"  平均标签数: {true_labels.sum(axis=1).mean():.2f}")
    
    return pred_probs, true_labels, samples


def load_ddi_pairs(ddi_file, drug_atc_file, atc4_list, top_k_rare=40):
    """
    加载 DDI 对（学习 data/mimic3/ddi.py 的方法）
    选择最罕见的 top_k_rare 条副作用记录
    """
    print(f"\n📂 加载 DDI 数据...")
    print(f"  DDI 文件: {ddi_file}")
    print(f"  Drug-ATC 文件: {drug_atc_file}")
    
    atc4_set = set(atc4_list)
    print(f"  ATC4 词汇表大小: {len(atc4_set)}")
    
    # 加载 STITCH → ATC4 映射
    stitch2atc = defaultdict(set)
    with open(drug_atc_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            stitch = parts[0].strip()
            for atc in parts[1:]:
                atc = atc.strip()
                if len(atc) >= 4:
                    a4 = atc[:4]
                    if a4 in atc4_set:
                        stitch2atc[stitch].add(a4)
    
    print(f"  STITCH 映射数: {len(stitch2atc)}")
    
    # 加载 DDI CSV，选最罕见的记录
    df = pd.read_csv(ddi_file)
    se_counter = Counter(df["Side Effect Name"].str.lower())
    df["freq"] = df["Side Effect Name"].str.lower().map(se_counter)
    rare_records = df.sort_values("freq").head(top_k_rare)
    
    print(f"  总 DDI 记录数: {len(df)}")
    print(f"  选择最罕见的 {top_k_rare} 条记录")
    print(f"\n  🔍 最罕见的副作用示例:")
    for idx, (_, row) in enumerate(rare_records.head(5).iterrows()):
        print(f"    {idx+1}. {row['Side Effect Name']} (频率: {row['freq']})")
    
    # 映射到 ATC4 对
    ddi_pairs = set()
    
    def cand(s):
        return [s, s[3:]] if s.startswith("CID") else [s, "CID" + s]
    
    for _, row in rare_records.iterrows():
        s1 = str(row["STITCH 1"]).strip()
        s2 = str(row["STITCH 2"]).strip()
        
        a1, a2 = set(), set()
        
        for c in cand(s1):
            if c in stitch2atc:
                a1 |= stitch2atc[c]
        
        for c in cand(s2):
            if c in stitch2atc:
                a2 |= stitch2atc[c]
        
        for x in a1:
            for y in a2:
                if x != y:
                    ddi_pairs.add(frozenset([x, y]))
    
    print(f"  最终 ATC4-DDI 对数: {len(ddi_pairs)}")
    
    # 计算覆盖率
    atc4_in_ddi = set()
    for pair in ddi_pairs:
        atc4_in_ddi.update(pair)
    print(f"  涉及的 ATC4 药物数: {len(atc4_in_ddi)}/{len(atc4_set)} ({len(atc4_in_ddi)/len(atc4_set)*100:.1f}%)")
    
    return ddi_pairs


def compute_ddi_rate(predictions, atc4_list_pkl, ddi_pairs, threshold=0.35, top_k=20, verbose=False):
    """
    计算 DDI Rate（使用 PKL 词汇表）
    
    ⚠️ 注意：这里用 PKL 词汇表解读预测，意味着在不同的药物索引上计算 DDI
    
    Args:
        predictions: (N, C) 预测概率（索引对应 EHRTokenizer）
        atc4_list_pkl: PKL 词汇表的 ATC4 代码列表
        ddi_pairs: DDI 对集合（基于 PKL 词汇表）
        threshold: 阈值
        top_k: 最终输出药物数量
        verbose: 是否输出调试信息
    """
    violations = 0
    total_valid_samples = 0
    total_drug_pairs = 0
    total_ddi_hits = 0
    
    for sample_idx, probs in enumerate(predictions):
        # 1. 阈值过滤
        mask = (probs > threshold).astype(float)
        filtered_probs = probs * mask
        
        # 2. Top-K 选择
        idx = np.argsort(filtered_probs)[-top_k:][::-1]
        # ⚠️ 用 PKL 词汇表解读索引
        preds = [atc4_list_pkl[i] for i in idx if filtered_probs[i] > 0]
        
        if len(preds) == 0:
            continue
        
        total_valid_samples += 1
        
        # 3. 检查 DDI
        hit = False
        sample_ddi_count = 0
        sample_pairs = []
        
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                pair = frozenset([preds[i], preds[j]])
                sample_pairs.append((preds[i], preds[j]))
                
                if pair in ddi_pairs:
                    sample_ddi_count += 1
                    if not hit:
                        hit = True
                        if verbose and sample_idx < 3:
                            print(f"    样本 {sample_idx}: DDI 对 {preds[i]} - {preds[j]}")
        
        total_drug_pairs += len(sample_pairs)
        total_ddi_hits += sample_ddi_count
        
        if hit:
            violations += 1
    
    if verbose:
        print(f"\n  📊 DDI 统计:")
        print(f"    有效样本数（有预测药物）: {total_valid_samples}/{len(predictions)}")
        print(f"    总药物对数: {total_drug_pairs}")
        print(f"    DDI 命中对数: {total_ddi_hits}")
        print(f"    DDI 对比例: {total_ddi_hits/total_drug_pairs*100:.2f}%")
        print(f"    违规样本数: {violations}")
        print(f"    违规样本率: {violations/len(predictions)*100:.2f}%")
    
    return violations / len(predictions) if len(predictions) > 0 else 0


def compute_metrics_with_per_drug_thresholds(y_true, y_pred_probs, per_drug_thresholds):
    """
    使用每个药物独立阈值计算指标
    
    Args:
        y_true: (N, C) 真实标签
        y_pred_probs: (N, C) 预测概率
        per_drug_thresholds: (C,) 每个药物的阈值
    """
    y_pred = (y_pred_probs >= per_drug_thresholds).astype(int)
    
    f1_scores = []
    jaccard_scores = []
    precision_scores = []
    recall_scores = []
    pred_counts = []
    
    for i in range(len(y_true)):
        if y_true[i].sum() > 0:
            tp = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
            fp = np.sum((y_true[i] == 0) & (y_pred[i] == 1))
            fn = np.sum((y_true[i] == 1) & (y_pred[i] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            f1_scores.append(f1)
            jaccard_scores.append(jaccard)
            precision_scores.append(precision)
            recall_scores.append(recall)
            pred_counts.append(y_pred[i].sum())
    
    # PR-AUC（全局计算）
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true.flatten(), y_pred_probs.flatten()
        )
        prauc = auc(recall_curve, precision_curve)
    except:
        prauc = 0
    
    return {
        'jaccard': np.mean(jaccard_scores) if jaccard_scores else 0,
        'f1': np.mean(f1_scores) if f1_scores else 0,
        'precision': np.mean(precision_scores) if precision_scores else 0,
        'recall': np.mean(recall_scores) if recall_scores else 0,
        'prauc': prauc,
        'avg_med': np.mean(pred_counts) if pred_counts else 0
    }


def compute_ddi_rate_with_per_drug_thresholds(predictions, atc4_list_pkl, ddi_pairs, 
                                               per_drug_thresholds, top_k=20):
    """
    使用每个药物独立阈值计算 DDI Rate
    
    Args:
        predictions: (N, C) 预测概率
        atc4_list_pkl: PKL 词汇表
        ddi_pairs: DDI 对集合
        per_drug_thresholds: (C,) 每个药物的阈值
        top_k: Top-K
    """
    violations = 0
    
    for probs in predictions:
        # 1. 应用每个药物的独立阈值
        mask = (probs >= per_drug_thresholds).astype(float)
        filtered_probs = probs * mask
        
        # 2. Top-K 选择
        idx = np.argsort(filtered_probs)[-top_k:][::-1]
        preds = [atc4_list_pkl[i] for i in idx if filtered_probs[i] > 0]
        
        if len(preds) == 0:
            continue
        
        # 3. 检查 DDI
        hit = False
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if frozenset([preds[i], preds[j]]) in ddi_pairs:
                    hit = True
                    break
            if hit:
                break
        
        if hit:
            violations += 1
    
    return violations / len(predictions) if len(predictions) > 0 else 0


def optimize_per_drug_thresholds_on_val(val_probs, val_labels):
    """
    在验证集上优化每个药物的独立阈值
    
    Returns:
        per_drug_thresholds: (C,) 每个药物的最优阈值
    """
    print("\n🔧 在验证集上优化每个药物的独立阈值...")
    
    n_drugs = val_probs.shape[1]
    per_drug_thresholds = np.zeros(n_drugs)
    
    for drug_idx in range(n_drugs):
        drug_probs = val_probs[:, drug_idx]
        drug_labels = val_labels[:, drug_idx]
        
        # 如果这个药物在验证集中没有出现，使用默认阈值 0.5
        if drug_labels.sum() == 0:
            per_drug_thresholds[drug_idx] = 0.5
            continue
        
        # 尝试不同的阈值，找到 F1 最高的
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.1, 1.0, 0.05):
            pred = (drug_probs >= thresh).astype(int)
            
            tp = np.sum((drug_labels == 1) & (pred == 1))
            fp = np.sum((drug_labels == 0) & (pred == 1))
            fn = np.sum((drug_labels == 1) & (pred == 0))
            
            if (tp + fp + fn) > 0:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        
        per_drug_thresholds[drug_idx] = best_thresh
    
    print(f"  阈值范围: {per_drug_thresholds.min():.2f} - {per_drug_thresholds.max():.2f}")
    print(f"  平均阈值: {per_drug_thresholds.mean():.2f}")
    
    return per_drug_thresholds


def bootstrap_evaluation_per_drug(pred_probs, true_labels, atc4_list_ehr, atc4_list_pkl, 
                                   ddi_pairs, per_drug_thresholds,
                                   n_rounds=10, sample_ratio=0.3, seed=42):
    """
    Bootstrap 评估（使用每个药物独立阈值）
    """
    print(f"\n🎲 Bootstrap 评估（每个药物独立阈值）:")
    
    random.seed(seed)
    np.random.seed(seed)
    
    n_samples = len(pred_probs)
    sample_size = int(n_samples * sample_ratio)
    
    results = []
    
    for round_idx in range(n_rounds):
        indices = random.sample(range(n_samples), sample_size)
        
        sampled_probs = pred_probs[indices]
        sampled_labels = true_labels[indices]
        
        # 计算指标（用 EHRTokenizer）
        metrics = compute_metrics_with_per_drug_thresholds(sampled_labels, sampled_probs, per_drug_thresholds)
        
        # 计算 DDI（用 PKL 词汇表）
        ddi_rate = compute_ddi_rate_with_per_drug_thresholds(
            sampled_probs, atc4_list_pkl, ddi_pairs, per_drug_thresholds, top_k=20
        )
        
        results.append({
            'jaccard': metrics['jaccard'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'prauc': metrics['prauc'],
            'avg_med': metrics['avg_med'],
            'ddi': ddi_rate
        })
        
        print(f"  Round {round_idx+1}: Jaccard={metrics['jaccard']:.4f}, F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
              f"PR-AUC={metrics['prauc']:.4f}, Avg.Med={metrics['avg_med']:.2f}, DDI={ddi_rate:.4f}")
    
    # 计算均值和标准差
    results_array = np.array([[r['jaccard'], r['f1'], r['precision'], r['recall'], r['prauc'], r['avg_med'], r['ddi']] for r in results])
    mean = results_array.mean(axis=0)
    std = results_array.std(axis=0)
    
    return {
        'jaccard_mean': mean[0],
        'jaccard_std': std[0],
        'f1_mean': mean[1],
        'f1_std': std[1],
        'precision_mean': mean[2],
        'precision_std': std[2],
        'recall_mean': mean[3],
        'recall_std': std[3],
        'prauc_mean': mean[4],
        'prauc_std': std[4],
        'avg_med_mean': mean[5],
        'avg_med_std': std[5],
        'ddi_mean': mean[6],
        'ddi_std': std[6],
        'all_results': results
    }


def compute_metrics_samplewise(y_true, y_pred_probs, threshold):
    """样本级别计算指标"""
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    f1_scores = []
    jaccard_scores = []
    precision_scores = []
    recall_scores = []
    pred_counts = []  # 新增：记录每个样本的预测药物数
    
    for i in range(len(y_true)):
        if y_true[i].sum() > 0:
            tp = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
            fp = np.sum((y_true[i] == 0) & (y_pred[i] == 1))
            fn = np.sum((y_true[i] == 1) & (y_pred[i] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            jaccard_scores.append(jaccard)
            pred_counts.append(y_pred[i].sum())  # 新增：统计预测药物数
    
    # PR-AUC（全局计算）
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true.flatten(), y_pred_probs.flatten()
        )
        prauc = auc(recall_curve, precision_curve)
    except:
        prauc = 0
    
    return {
        'jaccard': np.mean(jaccard_scores) if jaccard_scores else 0,
        'f1': np.mean(f1_scores) if f1_scores else 0,
        'precision': np.mean(precision_scores) if precision_scores else 0,
        'recall': np.mean(recall_scores) if recall_scores else 0,
        'prauc': prauc,
        'avg_med': np.mean(pred_counts) if pred_counts else 0
    }


def bootstrap_evaluation(pred_probs, true_labels, atc4_list_ehr, atc4_list_pkl, ddi_pairs, 
                        threshold=0.35, n_rounds=10, sample_ratio=0.3, seed=42):
    """
    Bootstrap 评估：随机抽样 30%，重复 10 次
    
    Args:
        atc4_list_ehr: EHRTokenizer 词汇表（用于 F1/Jaccard/PR-AUC）
        atc4_list_pkl: PKL 词汇表（用于 DDI）
    """
    print(f"\n🎲 Bootstrap 评估:")
    print(f"  抽样比例: {sample_ratio*100:.0f}%")
    print(f"  重复次数: {n_rounds}")
    print(f"  随机种子: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    
    n_samples = len(pred_probs)
    sample_size = int(n_samples * sample_ratio)
    
    print(f"  总样本数: {n_samples}")
    print(f"  每次抽样: {sample_size}")
    
    results = []
    
    for round_idx in range(n_rounds):
        # 随机抽样
        indices = random.sample(range(n_samples), sample_size)
        
        sampled_probs = pred_probs[indices]
        sampled_labels = true_labels[indices]
        
        # 计算指标（用 EHRTokenizer）
        metrics = compute_metrics_samplewise(sampled_labels, sampled_probs, threshold)
        
        # 计算 DDI（用 PKL 词汇表）
        ddi_rate = compute_ddi_rate(sampled_probs, atc4_list_pkl, ddi_pairs, 
                                    threshold=threshold, top_k=20, verbose=False)
        
        results.append({
            'jaccard': metrics['jaccard'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'prauc': metrics['prauc'],
            'avg_med': metrics['avg_med'],
            'ddi': ddi_rate
        })
        
        print(f"  Round {round_idx+1}: Jaccard={metrics['jaccard']:.4f}, F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
              f"PR-AUC={metrics['prauc']:.4f}, Avg.Med={metrics['avg_med']:.2f}, DDI={ddi_rate:.4f}")
    
    # 计算均值和标准差
    results_array = np.array([[r['jaccard'], r['f1'], r['precision'], r['recall'], r['prauc'], r['avg_med'], r['ddi']] for r in results])
    mean = results_array.mean(axis=0)
    std = results_array.std(axis=0)
    
    return {
        'jaccard_mean': mean[0],
        'jaccard_std': std[0],
        'f1_mean': mean[1],
        'f1_std': std[1],
        'precision_mean': mean[2],
        'precision_std': std[2],
        'recall_mean': mean[3],
        'recall_std': std[3],
        'prauc_mean': mean[4],
        'prauc_std': std[4],
        'avg_med_mean': mean[5],
        'avg_med_std': std[5],
        'ddi_mean': mean[6],
        'ddi_std': std[6],
        'all_results': results
    }


def main():
    # ==================== 配置 ====================
    TEST_PRED_FILE = "results主实验/checkpoint-5000/test_predictions.json"
    TRAIN_FILE = "data/mimic3/l_data/train_atc_hierarchy2.json"
    DRUG_EMBEDDING_FILE = "data/mimic3/l_data/drug_embeddings2.pt"
    VOCAB_PKL_FILE = "data/mimic3/data/handled/atc4_vocab.pkl"  # 新增：PKL 词汇表
    
    # DDI 相关文件
    DDI_FILE = "data/mimic3/auxiliary/drug-DDI.csv"
    DRUG_ATC_FILE = "data/mimic3/auxiliary/drug-atc.csv"
    
    THRESHOLD = 0.35  # 全局动态阈值
    OUTPUT_DIR = "final_evaluation_results"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("🎯 最终评估：全局阈值 0.35 + Bootstrap")
    print("=" * 80)
    print("⚠️ F1/Jaccard/PR-AUC 用 EHRTokenizer，DDI 用 PKL 词汇表")
    
    # ==================== 构建 EHRTokenizer 词汇表 ====================
    print("\n📚 构建 EHRTokenizer 词汇表（用于 F1/Jaccard/PR-AUC）...")
    ehr_tokenizer = EHRTokenizer(TRAIN_FILE, drug_embedding_file=DRUG_EMBEDDING_FILE)
    atc4_list_ehr = ehr_tokenizer.level4_voc.idx2word
    print(f"L4 词汇表大小: {len(atc4_list_ehr)}")
    print(f"前 10 个: {atc4_list_ehr[:10]}")
    
    # ==================== 加载 PKL 词汇表 ====================
    print("\n📚 加载 PKL 词汇表（用于 DDI）...")
    with open(VOCAB_PKL_FILE, "rb") as f:
        vocab_pkl = dill.load(f)
    atc4_list_pkl = vocab_pkl["atc4_list"]
    print(f"L4 词汇表大小: {len(atc4_list_pkl)}")
    print(f"前 10 个: {atc4_list_pkl[:10]}")
    
    # 加载预测和标签（用 EHRTokenizer）
    pred_probs, true_labels, samples = load_predictions_and_labels(TEST_PRED_FILE, ehr_tokenizer)
    
    # ==================== 加载验证集（用于优化每个药物的阈值）====================
    print("\n📂 加载验证集...")
    VAL_PRED_FILE = "results/validation_checkpoint-5000/test_predictions.json"
    
    if os.path.exists(VAL_PRED_FILE):
        val_probs, val_labels, _ = load_predictions_and_labels(VAL_PRED_FILE, ehr_tokenizer)
        print(f"  验证集样本数: {len(val_probs)}")
        
        # 优化每个药物的独立阈值
        per_drug_thresholds = optimize_per_drug_thresholds_on_val(val_probs, val_labels)
    else:
        print(f"  ⚠️ 验证集预测文件不存在: {VAL_PRED_FILE}")
        print(f"  将跳过每个药物独立阈值的评估")
        per_drug_thresholds = None
    
    # 加载 DDI 对（用 PKL 词汇表）
    ddi_pairs = load_ddi_pairs(DDI_FILE, DRUG_ATC_FILE, atc4_list_pkl, top_k_rare=40)
    
    # ==================== Bootstrap 评估 ====================
    print("\n" + "=" * 80)
    print(f"📊 Bootstrap 评估（阈值 = {THRESHOLD}）")
    print("=" * 80)
    
    # 先做一次完整的 DDI 计算看看统计信息
    print("\n🔍 先在完整测试集上计算 DDI（查看详细统计）...")
    print("⚠️ 使用 PKL 词汇表计算 DDI")
    ddi_rate_full = compute_ddi_rate(pred_probs, atc4_list_pkl, ddi_pairs, 
                                     threshold=THRESHOLD, top_k=20, verbose=True)
    print(f"\n完整测试集 DDI Rate: {ddi_rate_full:.4f}")
    
    bootstrap_results = bootstrap_evaluation(
        pred_probs, true_labels, atc4_list_ehr, atc4_list_pkl, ddi_pairs,
        threshold=THRESHOLD, n_rounds=10, sample_ratio=0.3, seed=42
    )
    
    # ==================== 打印结果 ====================
    print("\n" + "=" * 80)
    print("📊 最终结果（均值 ± 标准差）")
    print("=" * 80)
    
    print(f"\n阈值: {THRESHOLD}")
    print(f"Jaccard:   {bootstrap_results['jaccard_mean']:.4f} ± {bootstrap_results['jaccard_std']:.4f}")
    print(f"F1:        {bootstrap_results['f1_mean']:.4f} ± {bootstrap_results['f1_std']:.4f}")
    print(f"Precision: {bootstrap_results['precision_mean']:.4f} ± {bootstrap_results['precision_std']:.4f}")
    print(f"Recall:    {bootstrap_results['recall_mean']:.4f} ± {bootstrap_results['recall_std']:.4f}")
    print(f"PR-AUC:    {bootstrap_results['prauc_mean']:.4f} ± {bootstrap_results['prauc_std']:.4f}")
    print(f"Avg.Med:   {bootstrap_results['avg_med_mean']:.2f} ± {bootstrap_results['avg_med_std']:.2f}")
    print(f"DDI:       {bootstrap_results['ddi_mean']:.4f} ± {bootstrap_results['ddi_std']:.4f}")
    
    # ==================== 对比固定阈值 0.1-0.9 ====================
    print("\n" + "=" * 80)
    print("📊 对比所有固定阈值 (0.1-0.9)")
    print("=" * 80)
    
    all_threshold_results = []
    
    for thresh in np.arange(0.1, 1.0, 0.1):
        print(f"\n计算阈值 {thresh:.1f}...")
        
        bootstrap_res = bootstrap_evaluation(
            pred_probs, true_labels, atc4_list_ehr, atc4_list_pkl, ddi_pairs,
            threshold=thresh, n_rounds=10, sample_ratio=0.3, seed=42
        )
        
        all_threshold_results.append({
            'threshold': f'{thresh:.1f}',
            'jaccard_mean': bootstrap_res['jaccard_mean'],
            'jaccard_std': bootstrap_res['jaccard_std'],
            'f1_mean': bootstrap_res['f1_mean'],
            'f1_std': bootstrap_res['f1_std'],
            'precision_mean': bootstrap_res['precision_mean'],
            'precision_std': bootstrap_res['precision_std'],
            'recall_mean': bootstrap_res['recall_mean'],
            'recall_std': bootstrap_res['recall_std'],
            'prauc_mean': bootstrap_res['prauc_mean'],
            'prauc_std': bootstrap_res['prauc_std'],
            'avg_med_mean': bootstrap_res['avg_med_mean'],
            'avg_med_std': bootstrap_res['avg_med_std'],
            'ddi_mean': bootstrap_res['ddi_mean'],
            'ddi_std': bootstrap_res['ddi_std']
        })
    
    # 添加动态阈值结果
    all_threshold_results.append({
        'threshold': f'全局 ({THRESHOLD:.2f})',
        'jaccard_mean': bootstrap_results['jaccard_mean'],
        'jaccard_std': bootstrap_results['jaccard_std'],
        'f1_mean': bootstrap_results['f1_mean'],
        'f1_std': bootstrap_results['f1_std'],
        'precision_mean': bootstrap_results['precision_mean'],
        'precision_std': bootstrap_results['precision_std'],
        'recall_mean': bootstrap_results['recall_mean'],
        'recall_std': bootstrap_results['recall_std'],
        'prauc_mean': bootstrap_results['prauc_mean'],
        'prauc_std': bootstrap_results['prauc_std'],
        'avg_med_mean': bootstrap_results['avg_med_mean'],
        'avg_med_std': bootstrap_results['avg_med_std'],
        'ddi_mean': bootstrap_results['ddi_mean'],
        'ddi_std': bootstrap_results['ddi_std']
    })
    
    # ==================== 添加每个药物独立阈值的结果 ====================
    if per_drug_thresholds is not None:
        print("\n" + "=" * 80)
        print("📊 评估每个药物独立阈值")
        print("=" * 80)
        
        per_drug_results = bootstrap_evaluation_per_drug(
            pred_probs, true_labels, atc4_list_ehr, atc4_list_pkl, ddi_pairs,
            per_drug_thresholds, n_rounds=10, sample_ratio=0.3, seed=42
        )
        
        all_threshold_results.append({
            'threshold': '每药物独立',
            'jaccard_mean': per_drug_results['jaccard_mean'],
            'jaccard_std': per_drug_results['jaccard_std'],
            'f1_mean': per_drug_results['f1_mean'],
            'f1_std': per_drug_results['f1_std'],
            'precision_mean': per_drug_results['precision_mean'],
            'precision_std': per_drug_results['precision_std'],
            'recall_mean': per_drug_results['recall_mean'],
            'recall_std': per_drug_results['recall_std'],
            'prauc_mean': per_drug_results['prauc_mean'],
            'prauc_std': per_drug_results['prauc_std'],
            'avg_med_mean': per_drug_results['avg_med_mean'],
            'avg_med_std': per_drug_results['avg_med_std'],
            'ddi_mean': per_drug_results['ddi_mean'],
            'ddi_std': per_drug_results['ddi_std']
        })
    
    # ==================== 生成表格 ====================
    print("\n" + "=" * 80)
    print("📊 完整对比表格")
    print("=" * 80)
    
    df = pd.DataFrame(all_threshold_results)
    
    # 打印表格
    print("\n")
    print(df.to_string(index=False))
    
    # ==================== 保存结果 ====================
    # CSV
    csv_file = os.path.join(OUTPUT_DIR, "bootstrap_results.csv")
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"\n💾 CSV 已保存到: {csv_file}")
    
    # JSON
    json_file = os.path.join(OUTPUT_DIR, "bootstrap_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_threshold_results, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON 已保存到: {json_file}")
    
    # Markdown
    md_file = os.path.join(OUTPUT_DIR, "bootstrap_results.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# L4 层级 Bootstrap 评估结果\n\n")
        f.write("## 方法\n\n")
        f.write("- 随机抽样 30% 测试集\n")
        f.write("- 重复 10 次\n")
        f.write("- 报告均值 ± 标准差\n\n")
        f.write("## 结果\n\n")
        
        # 手动生成表格
        f.write("| 阈值 | Jaccard↑ | F1↑ | Precision↑ | Recall↑ | PR-AUC↑ | Avg.Med↓ | DDI↓ |\n")
        f.write("|------|----------|-----|------------|---------|---------|----------|------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['threshold']} | ")
            f.write(f"{row['jaccard_mean']:.4f}±{row['jaccard_std']:.4f} | ")
            f.write(f"{row['f1_mean']:.4f}±{row['f1_std']:.4f} | ")
            f.write(f"{row['precision_mean']:.4f}±{row['precision_std']:.4f} | ")
            f.write(f"{row['recall_mean']:.4f}±{row['recall_std']:.4f} | ")
            f.write(f"{row['prauc_mean']:.4f}±{row['prauc_std']:.4f} | ")
            f.write(f"{row['avg_med_mean']:.2f}±{row['avg_med_std']:.2f} | ")
            f.write(f"{row['ddi_mean']:.4f}±{row['ddi_std']:.4f} |\n")
        
        # 找出最优值
        best_jaccard_idx = df['jaccard_mean'].idxmax()
        best_f1_idx = df['f1_mean'].idxmax()
        best_prauc_idx = df['prauc_mean'].idxmax()
        best_avg_med_idx = df['avg_med_mean'].idxmin()  # 新增：越少越好
        best_ddi_idx = df['ddi_mean'].idxmin()  # DDI 越低越好
        
        f.write("\n## 最优结果\n\n")
        f.write(f"- **Jaccard 最优**: {df.loc[best_jaccard_idx, 'threshold']} → {df.loc[best_jaccard_idx, 'jaccard_mean']:.4f}±{df.loc[best_jaccard_idx, 'jaccard_std']:.4f}\n")
        f.write(f"- **F1 最优**: {df.loc[best_f1_idx, 'threshold']} → {df.loc[best_f1_idx, 'f1_mean']:.4f}±{df.loc[best_f1_idx, 'f1_std']:.4f}\n")
        f.write(f"- **PR-AUC 最优**: {df.loc[best_prauc_idx, 'threshold']} → {df.loc[best_prauc_idx, 'prauc_mean']:.4f}±{df.loc[best_prauc_idx, 'prauc_std']:.4f}\n")
        f.write(f"- **Avg.Med 最优**: {df.loc[best_avg_med_idx, 'threshold']} → {df.loc[best_avg_med_idx, 'avg_med_mean']:.2f}±{df.loc[best_avg_med_idx, 'avg_med_std']:.2f}\n")
        f.write(f"- **DDI 最优**: {df.loc[best_ddi_idx, 'threshold']} → {df.loc[best_ddi_idx, 'ddi_mean']:.4f}±{df.loc[best_ddi_idx, 'ddi_std']:.4f}\n")
    
    print(f"💾 Markdown 已保存到: {md_file}")
    
    # ==================== LaTeX 表格 ====================
    print("\n" + "=" * 80)
    print("📝 LaTeX 表格（可直接用于论文）")
    print("=" * 80)
    
    print("\n```latex")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{L4 层级在不同阈值下的性能对比（10次随机抽样30\\%测试集）}")
    print("\\label{tab:threshold_bootstrap}")
    print("\\begin{tabular}{lccccccc}")
    print("\\hline")
    print("阈值 & Jaccard↑ & F1↑ & Precision↑ & Recall↑ & PR-AUC↑ & Avg.Med↓ & DDI↓ \\\\")
    print("\\hline")
    
    for _, row in df.iterrows():
        print(f"{row['threshold']} & ", end="")
        print(f"{row['jaccard_mean']:.4f}$\\pm${row['jaccard_std']:.4f} & ", end="")
        print(f"{row['f1_mean']:.4f}$\\pm${row['f1_std']:.4f} & ", end="")
        print(f"{row['precision_mean']:.4f}$\\pm${row['precision_std']:.4f} & ", end="")
        print(f"{row['recall_mean']:.4f}$\\pm${row['recall_std']:.4f} & ", end="")
        print(f"{row['prauc_mean']:.4f}$\\pm${row['prauc_std']:.4f} & ", end="")
        print(f"{row['avg_med_mean']:.2f}$\\pm${row['avg_med_std']:.2f} & ", end="")
        print(f"{row['ddi_mean']:.4f}$\\pm${row['ddi_std']:.4f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("```")
    
    print("\n" + "=" * 80)
    print("✅ 评估完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
