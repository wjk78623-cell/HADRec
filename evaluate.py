# evaluate.py
# ✅ 保留原始计算逻辑，但不再加载 voc_final.pkl
# ✅ ehr_tokenizer 由 main_llm_cls.py 传入

import copy
import os
import pickle
import numpy as np
from utils.utils import read_jsonlines, multi_label_metric, ddi_rate_score


def np_sigmoid(x):
    """Sigmoid 函数（Numpy实现）"""
    return 1 / (1 + np.exp(-x))


def evaluate_jsonlines(data_path, ehr_tokenizer, threshold=0.4, ddi_path='./data/handled/'):
    """
    评估预测结果（使用L4层级词汇表，与层级独立评估保持一致）

    参数:
    ----------
    data_path : str  
        预测结果 JSON 文件路径（如 ./results/test_predictions.json）
    ehr_tokenizer : EHRTokenizer  
        来自主程序的 tokenizer（包含 level4_voc.word2idx）
    threshold : float  
        二值化阈值（默认 0.4，与层级独立评估保持一致）
    ddi_path : str  
        DDI 邻接矩阵路径（含 ddi_A_final.pkl）

    返回:
    ----------
    (ja, prauc, avg_p, avg_r, avg_f1)
    """

    # 初始化存储变量（使用L4词汇表）
    pred_data_prob, pred_data = [], []
    true_data = np.zeros((len(read_jsonlines(data_path)), len(ehr_tokenizer.level4_voc.word2idx)))
    pred_label = []

    # 遍历每条预测结果
    for row, meta_data in enumerate(read_jsonlines(data_path)):

        # normalize the predicted scores by sigmoid, and get the prob
        meta_pred_data_prob = np.array(meta_data["target"])
        pred_data_prob.append(np_sigmoid(meta_pred_data_prob))

        # transform y to 0-1 by threshold
        meta_pred_data = copy.deepcopy(np_sigmoid(meta_pred_data_prob))
        meta_pred_data[meta_pred_data >= threshold] = 1
        meta_pred_data[meta_pred_data < threshold] = 0
        pred_data.append(meta_pred_data)

        # get the true data
        if "drug_code" in meta_data:
            true_index = ehr_tokenizer.convert_med_tokens_to_ids(meta_data["drug_code"])
            true_data[row][true_index] = 1

        # prepare the labels for DDI calculation
        # ✅ 修复：使用正确的3层嵌套格式 [[就诊1], [就诊2], ...]
        # 其中每个就诊是一个药物ID列表
        meta_label = np.where(meta_pred_data == 1)[0]
        pred_label.append([sorted(meta_label)])  # [患者] -> [[就诊]] -> [药物ID列表]

    # === 计算多标签指标 ===
    ja, prauc, avg_p, avg_r, avg_f1, mean, std = multi_label_metric(
        true_data,
        np.array(pred_data),
        np.array(pred_data_prob)
    )

    # === 加载 DDI 邻接矩阵 ===
    ddi_file = os.path.join(ddi_path, 'ddi_A_final.pkl')
    if os.path.exists(ddi_file):
        ddi_adj = pickle.load(open(ddi_file, 'rb'))
        
        # 🔥 关键修复：创建索引映射
        # DDI矩阵是基于按字母顺序排序的ATC4代码构建的（见data_process.py）
        # 而ehr_tokenizer.level4_voc是按出现顺序构建的
        # 需要通过ATC4代码名称来映射索引
        
        # 获取ehr_tokenizer的词表（按出现顺序）
        tokenizer_voc = ehr_tokenizer.level4_voc
        tokenizer_atc4_list = tokenizer_voc.idx2word  # [ATC4_code1, ATC4_code2, ...]
        
        # 构建DDI矩阵对应的词表（按字母顺序，与data_process.py中的构建逻辑一致）
        # 注意：这里假设DDI矩阵是按字母顺序构建的，与data_process.py中的逻辑一致
        # 如果DDI矩阵的维度与tokenizer词表大小一致，假设它们对应相同的ATC4代码集合
        # 如果维度不一致，使用tokenizer中的ATC4代码集合，按字母顺序排序
        if len(tokenizer_atc4_list) == ddi_adj.shape[0]:
            # 维度一致：假设对应相同的ATC4代码集合，只是顺序不同
            # 按字母顺序排序tokenizer中的ATC4代码，构建映射
            ddi_atc4_list = sorted(tokenizer_atc4_list)
        else:
            # 维度不一致：使用tokenizer中的ATC4代码集合，按字母顺序排序
            # 注意：这可能导致某些DDI矩阵中的ATC4代码无法映射
            ddi_atc4_list = sorted(set(tokenizer_atc4_list))
            if len(ddi_atc4_list) != ddi_adj.shape[0]:
                print(f"⚠️ 警告：DDI矩阵维度 ({ddi_adj.shape[0]}) 与预期词表大小 ({len(ddi_atc4_list)}) 不一致")
                print(f"   将使用tokenizer中的ATC4代码子集进行映射")
        
        # 创建从tokenizer索引到DDI矩阵索引的映射
        tokenizer_to_ddi_idx = {}
        ddi_voc_word2idx = {atc4: idx for idx, atc4 in enumerate(ddi_atc4_list)}
        
        for tokenizer_idx, atc4_code in enumerate(tokenizer_atc4_list):
            if atc4_code in ddi_voc_word2idx:
                ddi_idx = ddi_voc_word2idx[atc4_code]
                # 检查索引是否在有效范围内
                if 0 <= ddi_idx < ddi_adj.shape[0]:
                    tokenizer_to_ddi_idx[tokenizer_idx] = ddi_idx
                else:
                    tokenizer_to_ddi_idx[tokenizer_idx] = -1
            else:
                # 如果tokenizer中的ATC4代码不在DDI矩阵词表中，标记为-1
                tokenizer_to_ddi_idx[tokenizer_idx] = -1
        
        # 将pred_label中的索引从tokenizer索引映射到DDI矩阵索引
        pred_label_ddi = []
        for patient in pred_label:
            patient_ddi = []
            for adm in patient:
                adm_ddi = []
                for med_idx in adm:
                    if med_idx in tokenizer_to_ddi_idx:
                        ddi_idx = tokenizer_to_ddi_idx[med_idx]
                        if ddi_idx >= 0:  # 只保留在DDI矩阵中的药物
                            adm_ddi.append(ddi_idx)
                if adm_ddi:  # 只添加非空的就诊
                    patient_ddi.append(sorted(adm_ddi))
            if patient_ddi:  # 只添加非空的患者
                pred_label_ddi.append(patient_ddi)
        
        # 调试信息
        print(f"\n🔍 DDI 调试信息:")
        print(f"  ehr_tokenizer.level4_voc 大小: {len(tokenizer_voc.word2idx)}")
        print(f"  DDI矩阵词表大小: {len(ddi_atc4_list)}")
        print(f"  ddi_adj 形状: {ddi_adj.shape}")
        print(f"  ✅ 词表对齐检查: {len(ddi_atc4_list) == ddi_adj.shape[0]}")
        print(f"  有效映射数量: {sum(1 for v in tokenizer_to_ddi_idx.values() if v >= 0)} / {len(tokenizer_to_ddi_idx)}")
        print(f"  pred_label 长度: {len(pred_label)}")
        print(f"  pred_label_ddi 长度: {len(pred_label_ddi)}")
        if len(pred_label_ddi) > 0:
            print(f"  pred_label_ddi[0] 类型: {type(pred_label_ddi[0])}, 值: {pred_label_ddi[0]}")
            if len(pred_label_ddi[0]) > 0:
                print(f"  pred_label_ddi[0][0] 类型: {type(pred_label_ddi[0][0])}, 值: {pred_label_ddi[0][0][:5] if len(pred_label_ddi[0][0]) > 5 else pred_label_ddi[0][0]}")
        print(f"  ddi_adj 中1的个数: {np.sum(ddi_adj)}")
        print(f"  ddi_adj 中1的比例: {np.sum(ddi_adj) / (ddi_adj.shape[0] * ddi_adj.shape[1]):.4f}")
        
        # 使用映射后的pred_label_ddi计算DDI
        ddi = ddi_rate_score(pred_label_ddi, ddi_adj)
    else:
        ddi = -1
        print(f"⚠️ 未找到 DDI 文件: {ddi_file}")

    # === 打印结果 ===
    print('\n📊 Evaluation Results:')
    print('--------------------------------------------------')
    # 修复：将 {:.4} 改为 {:.4f}
    print('Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, DDI_rate: {:.4f}'.format(
        ja, prauc, avg_p, avg_r, avg_f1, ddi
    ))
    print('10-rounds PRAUC: {:.5f} ± {:.5f}'.format(mean[0], std[0]))
    print('10-rounds Jaccard: {:.5f} ± {:.5f}'.format(mean[1], std[1]))
    print('10-rounds F1-score: {:.5f} ± {:.5f}'.format(mean[2], std[2]))
    print('--------------------------------------------------')
    print('Note: Single/Multiple visit analysis is skipped (no visit count info).')

    return ja, prauc, avg_p, avg_r, avg_f1


if __name__ == "__main__":
    print("❌ 此脚本应由 main_llm_cls.py 调用，请勿单独运行。")