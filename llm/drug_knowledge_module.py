"""
🔬 药物分子知识模块（独立插件）

设计原则：
1. 独立模块，不侵入主模型
2. 可选启用/禁用
3. 处理多分子聚合（均值/最大/加权）
4. 安全：不泄露目标药物
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DrugKnowledgeModule(nn.Module):
    """
    药物分子知识融合模块
    
    功能：
    1. 加载预计算的药物嵌入（ChemBERTa编码）
    2. EHR ↔ Drug Cross-Attention
    3. 计算对齐损失（可选）
    """
    
    def __init__(
        self,
        hidden_size: int,
        drug_embedding_file: str = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual_scale: float = 0.5,
        align_weight: float = 0.1,
        enabled: bool = True,
        voc_l4=None  # ✅ 新增：接收 L4 词汇表用于过滤
    ):
        """
        Args:
            hidden_size: 模型的hidden size
            drug_embedding_file: 预计算的药物嵌入文件路径
            num_heads: Cross-Attention的头数
            dropout: Dropout比例
            residual_scale: 残差融合的缩放比例
            align_weight: 对齐损失的权重
            enabled: 是否启用药物知识
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.residual_scale = residual_scale
        self.align_weight = align_weight
        self.enabled = enabled
        
        if not self.enabled or drug_embedding_file is None:
            print("⚠️ 药物知识模块未启用")
            return
        
        # 加载药物嵌入
        if os.path.exists(drug_embedding_file):
            drug_data = torch.load(drug_embedding_file, map_location='cpu')
            drug_embeddings_full = drug_data["atc4_embeddings"]  # [N_drug_full, embed_dim]
            atc4_to_idx_full = drug_data.get("atc4_to_idx", {})
            
            # ✅ 如果提供了词汇表，过滤嵌入以匹配词汇表
            if voc_l4 is not None:
                # idx2word 可能是列表或字典，需要处理两种情况
                if isinstance(voc_l4.idx2word, dict):
                    vocab_atc4_codes = list(voc_l4.idx2word.values())
                else:
                    vocab_atc4_codes = voc_l4.idx2word  # 已经是列表
                
                filtered_embeddings = []
                filtered_atc4_to_idx = {}
                
                for new_idx, atc4_code in enumerate(vocab_atc4_codes):
                    if atc4_code in atc4_to_idx_full:
                        old_idx = atc4_to_idx_full[atc4_code]
                        filtered_embeddings.append(drug_embeddings_full[old_idx])
                        filtered_atc4_to_idx[atc4_code] = new_idx
                
                if len(filtered_embeddings) > 0:
                    drug_embeddings = torch.stack(filtered_embeddings)
                    self.atc4_to_idx = filtered_atc4_to_idx
                    print(f"✅ [DrugKnowledge] 过滤药物嵌入以匹配词汇表:")
                    print(f"  - 文件中总数: {len(atc4_to_idx_full)}")
                    print(f"  - 词汇表数量: {len(vocab_atc4_codes)}")
                    print(f"  - 最终使用: {len(filtered_embeddings)}")
                else:
                    print("⚠️ 过滤后无有效药物嵌入")
                    self.enabled = False
                    return
            else:
                drug_embeddings = drug_embeddings_full
                self.atc4_to_idx = atc4_to_idx_full
                print(f"⚠️ 未提供词汇表，使用全部药物嵌入: {drug_embeddings.shape}")
            
            self.register_buffer("drug_embeddings", drug_embeddings)
            print(f"✅ [DrugKnowledge] 加载药物嵌入: {drug_embeddings.shape}")
            print(f"  - 文件: {drug_embedding_file}")
        else:
            print(f"⚠️ 药物嵌入文件不存在: {drug_embedding_file}")
            self.enabled = False
            return
        
        # 投影层（将药物嵌入维度投影到模型hidden_size）
        drug_embed_dim = drug_embeddings.shape[1]
        if drug_embed_dim != hidden_size:
            self.drug_proj = nn.Linear(drug_embed_dim, hidden_size)
            print(f"  - 投影层: {drug_embed_dim} -> {hidden_size}")
        else:
            self.drug_proj = nn.Identity()
        
        # EHR ↔ Drug Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # 融合后的归一化层
        self.fusion_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, ehr_hidden, labels_l4=None):
        """
        融合EHR表示与药物知识
        
        Args:
            ehr_hidden: [B, H] - EHR的pooled hidden states
            labels_l4: [B, N_drug] - L4层的标签（训练时可选）
        
        Returns:
            fused_hidden: [B, H] - 融合后的表示
            align_loss: float - 对齐损失（如果提供labels）
        """
        if not self.enabled:
            return ehr_hidden, None
        
        device = ehr_hidden.device
        B = ehr_hidden.size(0)
        
        # 1. 准备药物知识库
        # ✅ 确保 drug_embeddings 在正确的设备上（ZeRO-3 兼容）
        drug_embs_input = self.drug_embeddings.to(device)
        drug_embs = self.drug_proj(drug_embs_input)  # [N_drug, H]
        drug_embs_batch = drug_embs.unsqueeze(0).expand(B, -1, -1)  # [B, N_drug, H]
        
        # 2. Cross-Attention: Query=EHR, Key/Value=Drug
        query = ehr_hidden.unsqueeze(1)  # [B, 1, H]
        attn_out, attn_weights = self.cross_attn(
            query, drug_embs_batch, drug_embs_batch
        )  # [B, 1, H], [B, 1, N_drug]
        drug_context = attn_out.squeeze(1)  # [B, H]
        
        # 3. 残差融合
        fused_hidden = self.fusion_norm(
            ehr_hidden + self.residual_scale * drug_context
        )
        
        # 4. 计算对齐损失（可选）
        align_loss = None
        if self.training and labels_l4 is not None:
            # 🔹 关键：只使用存在于药物嵌入中的标签
            # 过滤掉不在映射文件中的ATC4 code
            target_drug_mask = labels_l4.to(drug_embs.dtype)  # [B, N_drug] - 匹配 drug_embs 的数据类型
            
            # 检查是否有有效的目标药物（sum > 0）
            valid_samples = target_drug_mask.sum(dim=1) > 0  # [B]
            
            if valid_samples.any():
                # 只对有有效标签的样本计算损失
                valid_mask = target_drug_mask[valid_samples]  # [B_valid, N_drug]
                valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                
                # 目标药物的平均嵌入
                target_drug_emb = torch.matmul(valid_mask, drug_embs) / valid_count  # [B_valid, H]
                
                # MSE损失：让融合后的EHR表示接近目标药物嵌入
                valid_fused = fused_hidden[valid_samples]  # [B_valid, H]
                align_loss = F.mse_loss(valid_fused, target_drug_emb.detach())
        
        return fused_hidden, align_loss
    
    def get_attention_weights(self, ehr_hidden):
        """
        获取EHR与药物之间的注意力权重（用于可解释性分析）
        
        Args:
            ehr_hidden: [B, H]
        
        Returns:
            attn_weights: [B, N_drug]
            top_drugs: List[List[str]] - 每个样本的top-k药物
        """
        if not self.enabled:
            return None, None
        
        device = ehr_hidden.device
        B = ehr_hidden.size(0)
        
        # ✅ 确保 drug_embeddings 在正确的设备上（ZeRO-3 兼容）
        drug_embs_input = self.drug_embeddings.to(device)
        drug_embs = self.drug_proj(drug_embs_input)
        drug_embs_batch = drug_embs.unsqueeze(0).expand(B, -1, -1)
        
        query = ehr_hidden.unsqueeze(1)
        _, attn_weights = self.cross_attn(
            query, drug_embs_batch, drug_embs_batch
        )  # [B, 1, N_drug]
        
        attn_weights = attn_weights.squeeze(1)  # [B, N_drug]
        
        # 找到每个样本的top-5药物
        idx2atc = {v: k for k, v in self.atc4_to_idx.items()}
        top_drugs = []
        for b in range(B):
            top_k_indices = torch.topk(attn_weights[b], k=min(5, attn_weights.size(1))).indices
            top_k_drugs = [idx2atc.get(idx.item(), f"UNK_{idx.item()}") for idx in top_k_indices]
            top_drugs.append(top_k_drugs)
        
        return attn_weights, top_drugs


def filter_atc4_codes_by_mapping(atc4_list, drug_embedding_file):
    """
    过滤 ATC4 codes，只保留在药物嵌入文件中存在的
    
    Args:
        atc4_list: List[str] - 原始的 ATC4 code 列表
        drug_embedding_file: str - 药物嵌入文件路径
    
    Returns:
        filtered_list: List[str] - 过滤后的列表
        removed_count: int - 被过滤掉的数量
    """
    if not os.path.exists(drug_embedding_file):
        print(f"⚠️ 药物嵌入文件不存在: {drug_embedding_file}，不进行过滤")
        return atc4_list, 0
    
    drug_data = torch.load(drug_embedding_file, map_location='cpu')
    valid_atc4_set = set(drug_data.get("atc4_to_idx", {}).keys())
    
    filtered_list = [code for code in atc4_list if code in valid_atc4_set]
    removed_count = len(atc4_list) - len(filtered_list)
    
    if removed_count > 0:
        removed_codes = [code for code in atc4_list if code not in valid_atc4_set]
        print(f"⚠️ 过滤掉 {removed_count} 个不在映射中的 ATC4 code:")
        print(f"   {', '.join(removed_codes[:5])}{'...' if len(removed_codes) > 5 else ''}")
    
    return filtered_list, removed_count


def load_drug_knowledge_from_pkl(pkl_file, output_file=None, model_name="DeepChem/ChemBERTa-77M-MLM"):
    """
    从pkl文件加载ATC→SMILES映射，并预计算嵌入
    
    Args:
        pkl_file: pkl文件路径，格式为 {atc_code: {smiles1, smiles2, ...}}
        output_file: 输出文件路径（.pt）
        model_name: ChemBERTa模型名称
    
    Returns:
        drug_embeddings: [N_drug, embed_dim]
        atc4_to_idx: {atc_code: idx}
    """
    import pickle
    
    print(f"📂 从 {pkl_file} 加载ATC→SMILES映射...")
    with open(pkl_file, "rb") as f:
        atc_to_smiles = pickle.load(f)
    
    # 转换set为list，并确保所有SMILES都是字符串
    atc_to_smiles_list = {}
    for atc_code, smiles_set in atc_to_smiles.items():
        if isinstance(smiles_set, set):
            # 转换为列表并确保每个元素都是字符串
            atc_to_smiles_list[atc_code] = [str(s) for s in smiles_set]
        elif isinstance(smiles_set, list):
            atc_to_smiles_list[atc_code] = [str(s) for s in smiles_set]
        else:
            atc_to_smiles_list[atc_code] = [str(smiles_set)]
    
    print(f"✅ 加载了 {len(atc_to_smiles_list)} 个ATC code")
    
    # 调用内部函数进行编码
    drug_embeddings, atc4_to_idx = _build_drug_embeddings(
        atc_to_smiles_list,
        model_name=model_name,
        aggregation_method="mean"
    )
    
    if output_file:
        print(f"💾 保存到 {output_file}...")
        torch.save({
            "atc4_embeddings": drug_embeddings,
            "atc4_to_idx": atc4_to_idx,
            "atc4_to_smiles": atc_to_smiles_list,
            "metadata": {
                "num_drugs": len(atc4_to_idx),
                "embedding_dim": drug_embeddings.shape[1],
                "aggregation_method": "mean",
                "encoder_name": model_name
            }
        }, output_file)
    
    return drug_embeddings, atc4_to_idx


def _build_drug_embeddings(atc_to_smiles, model_name="DeepChem/ChemBERTa-77M-MLM", aggregation_method="mean"):
    """
    内部函数：使用ChemBERTa编码药物SMILES并聚合
    
    Args:
        atc_to_smiles: {atc_code: [smiles_list]}
        model_name: ChemBERTa模型名称
        aggregation_method: 聚合方法（"mean"）
    
    Returns:
        drug_embeddings: [N_drug, embed_dim]
        atc4_to_idx: {atc_code: idx}
    """
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    print(f"🤖 加载ChemBERTa模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ 模型加载完成，设备: {device}")
    
    # 编码所有药物
    atc4_to_idx = {}
    all_embeddings = []
    
    print(f"\n🔄 开始编码 {len(atc_to_smiles)} 个ATC code...")
    for idx, (atc_code, smiles_list) in enumerate(atc_to_smiles.items()):
        atc4_to_idx[atc_code] = idx
        
        # 编码所有SMILES
        smiles_embeddings = []
        for smiles in smiles_list:
            try:
                # 确保 SMILES 是字符串类型
                smiles_str = str(smiles) if not isinstance(smiles, str) else smiles
                
                inputs = tokenizer(smiles_str, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # 使用 [CLS] token 的输出作为分子嵌入
                    embedding = outputs.last_hidden_state[:, 0, :].cpu()
                    smiles_embeddings.append(embedding)
            except Exception as e:
                smiles_preview = str(smiles)[:30] if len(str(smiles)) > 30 else str(smiles)
                print(f"⚠️ 无法编码 {atc_code} 的SMILES: {smiles_preview}... - {e}")
                continue
        
        if len(smiles_embeddings) == 0:
            # 如果所有SMILES都失败，使用零向量
            print(f"⚠️ {atc_code} 所有SMILES编码失败，使用零向量")
            embed_dim = 384  # ChemBERTa-77M的默认维度
            drug_emb = torch.zeros(1, embed_dim)
        else:
            # 聚合多个SMILES的嵌入
            if aggregation_method == "mean":
                drug_emb = torch.cat(smiles_embeddings, dim=0).mean(dim=0, keepdim=True)
            else:
                raise ValueError(f"不支持的聚合方法: {aggregation_method}")
        
        all_embeddings.append(drug_emb)
        
        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx + 1}/{len(atc_to_smiles)}")
    
    # 堆叠所有嵌入
    drug_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"\n✅ 编码完成！")
    print(f"   - 药物数量: {len(atc4_to_idx)}")
    print(f"   - 嵌入维度: {drug_embeddings.shape[1]}")
    
    return drug_embeddings, atc4_to_idx

