import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class LlamaForMedRec(LlamaPreTrainedModel):
    """
    LlamaForMedRec with Soft-Mask Cross-Attention (方案 A).
    - Adjacent-layer cross-attn: L2 <- L1, L3 <- L2, L4 <- L3
    - Soft filtering: use cross_threshold to suppress low-prob features (no hard top-k)
    - Residual fusion: hidden <- LayerNorm(hidden + residual_scale * context)
    - Soft hierarchical masks for consistency loss
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = LlamaModel(config)

        # 层级词表大小
        self.med_voc_l1 = kwargs.pop("med_voc_l1", 100)
        self.med_voc_l2 = kwargs.pop("med_voc_l2", 200)
        self.med_voc_l3 = kwargs.pop("med_voc_l3", 300)
        self.med_voc_l4 = kwargs.pop("med_voc_l4", 400)

        # voc objects (optional)
        self.voc_l1 = kwargs.pop("voc_l1", None)
        self.voc_l2 = kwargs.pop("voc_l2", None)
        self.voc_l3 = kwargs.pop("voc_l3", None)
        self.voc_l4 = kwargs.pop("voc_l4", None)

        # 控制交叉注意力（方案A）
        self.use_cross_attention = kwargs.pop("use_cross_attention", False)
        # soft-mask 阈值（小：保留更多信息；大：更强抑制）
        self.cross_threshold = float(kwargs.pop("cross_threshold", 0.1))
        # cross context 注入到 hidden 的缩放比例（残差权重）
        self.cross_residual_scale = float(kwargs.pop("cross_residual_scale", 0.7))
        # 是否启用层级预测
        self.hierarchical_prediction = kwargs.pop("hierarchical_prediction", True)

        # 分类头
        self.cls_head_l1 = nn.Linear(config.hidden_size, self.med_voc_l1)
        self.cls_head_l2 = nn.Linear(config.hidden_size, self.med_voc_l2)
        self.cls_head_l3 = nn.Linear(config.hidden_size, self.med_voc_l3)
        self.cls_head_l4 = nn.Linear(config.hidden_size, self.med_voc_l4)

        # Cross-Attention 层（每层各自）
        # batch_first=True so inputs are (B, T, H)
        self.cross_attn_l2 = nn.MultiheadAttention(config.hidden_size, num_heads=4, dropout=0.1, batch_first=True)
        self.cross_attn_l3 = nn.MultiheadAttention(config.hidden_size, num_heads=4, dropout=0.1, batch_first=True)
        self.cross_attn_l4 = nn.MultiheadAttention(config.hidden_size, num_heads=4, dropout=0.1, batch_first=True)

        # 残差归一化层
        self.norm_l2 = nn.LayerNorm(config.hidden_size)
        self.norm_l3 = nn.LayerNorm(config.hidden_size)
        self.norm_l4 = nn.LayerNorm(config.hidden_size)

        # small projection to generate Q if needed (kept for extensibility)
        self.q_fine_proj = nn.Identity()
        
        # 🔬 药物分子知识模块（可选插件）
        use_drug_knowledge = kwargs.pop("use_drug_knowledge", False)
        drug_embedding_file = kwargs.pop("drug_embedding_file", None)
        if use_drug_knowledge:
            from llm.drug_knowledge_module import DrugKnowledgeModule
            self.drug_knowledge = DrugKnowledgeModule(
                hidden_size=config.hidden_size,
                drug_embedding_file=drug_embedding_file,
                enabled=True,
                voc_l4=self.voc_l4  # ✅ 传入词汇表用于过滤嵌入
            )
        else:
            self.drug_knowledge = None

        # logging counters
        self._forward_cnt = 0
        self.print_freq = int(kwargs.pop("print_freq", 200))

        self.post_init()

    # ---------------- Cross-Attention (soft-mask, normalized) ----------------
    def _apply_cross_attention(self, hidden, coarse_logits, coarse_weight, attn_layer, threshold=None):
        """
        hidden: (B, H) - pooled or current hidden for this level (query)
        coarse_logits: (B, N) - logits of the upstream coarse level
        coarse_weight: (N, H) - weight matrix of upstream classifier (used as embeddings)
        attn_layer: nn.MultiheadAttention instance to use for this level
        threshold: float, optional override for self.cross_threshold

        Returns:
            attn_entropy: float
            context: (B, H)
            attn_weights: (B, N) soft weights (after normalization)
        """
        device = hidden.device
        if threshold is None:
            threshold = self.cross_threshold

        B = hidden.size(0)
        # 1) probs and soft mask
        probs = torch.sigmoid(coarse_logits)  # (B, N)
        masked = probs.clone()
        masked[masked < threshold] = 0.0  # suppress low-confidence entries

        # 2) if a row becomes all-zero, revert to original probs for that row
        zero_rows = (masked.sum(dim=1) == 0)
        if zero_rows.any():
            masked[zero_rows, :] = probs[zero_rows, :]

        # 3) normalize masked probs into weights (attention-like)
        denom = masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = masked / denom  # (B, N)

        # 4) context vector: weighted sum over coarse_weight
        # coarse_weight: (N, H)
        # ✅ 确保数据类型一致（FP32）
        coarse_weight_fp32 = coarse_weight.to(device=device, dtype=weights.dtype)
        context = torch.matmul(weights, coarse_weight_fp32)  # (B, H)

        # 5) use MultiheadAttention for richer mixing: expand dims to (B, 1, H)
        query = (self.q_fine_proj(hidden)).unsqueeze(1)  # (B,1,H)
        key = context.unsqueeze(1)  # (B,1,H)
        value = key

        # call the given attention layer (batch_first=True)
        attn_out, attn_w = attn_layer(query, key, value)  # attn_out: (B,1,H), attn_w: (B, 1, 1)
        attn_out = attn_out.squeeze(1)  # (B, H)

        # compute entropy over weights as diagnostic (lower => more focused)
        p = weights.clamp(min=1e-8)
        ent = - (p * torch.log(p)).sum(dim=1)  # (B,)
        attn_entropy = float(ent.mean().detach().cpu().item())

        return attn_entropy, attn_out, weights

    # ---------------- Soft hierarchical masks ----------------
    def _generate_hierarchical_masks(self, logits_l1, logits_l2, logits_l3, logits_l4):
        """
        Produce soft masks in [0,1] for L2/L3/L4 based on upstream probabilities.
        Returns mask_l2, mask_l3, mask_l4 with same shapes as logits_l2/3/4.
        """
        # probabilities
        l1_p = torch.sigmoid(logits_l1)  # (B, N1)
        l2_p = torch.sigmoid(logits_l2)  # (B, N2)
        l3_p = torch.sigmoid(logits_l3)  # (B, N3)
        l4_p = torch.sigmoid(logits_l4)  # (B, N4)

        # compute upstream summary signals (per-sample scalar or per-class guidance)
        # Here we use upstream mean confidence as a lightweight guide
        l1_mean = l1_p.mean(dim=1, keepdim=True)  # (B,1)
        l2_mean = l2_p.mean(dim=1, keepdim=True)
        l3_mean = l3_p.mean(dim=1, keepdim=True)

        # soft masks: blend upstream mean and local probs (keeps shape of target logits)
        mask_l2 = 0.5 * l1_mean + 0.5 * l2_p  # (B, N2)
        mask_l3 = 0.5 * l2_mean + 0.5 * l3_p  # (B, N3)
        mask_l4 = 0.5 * l3_mean + 0.5 * l4_p  # (B, N4)

        # clamp into [0,1]
        mask_l2 = mask_l2.clamp(0.0, 1.0)
        mask_l3 = mask_l3.clamp(0.0, 1.0)
        mask_l4 = mask_l4.clamp(0.0, 1.0)

        return mask_l2, mask_l3, mask_l4

    # ---------------- Consistency loss (unchanged semantics, soft masks) ----------------
    def _compute_consistency_loss(self, logits_l1, logits_l2, logits_l3, logits_l4, mask_l2, mask_l3, mask_l4):
        """
        Penalize logits that are active where mask is low.
        mask_l* are soft in [0,1] now; we scale penalty by (1-mask).
        """
        consistency_loss = 0.0
        if mask_l2 is not None:
            # penalize positions where mask ~ 0 but logits high
            consistency_loss += torch.mean(torch.relu(logits_l2 * (1 - mask_l2))) * 2.0
        if mask_l3 is not None:
            consistency_loss += torch.mean(torch.relu(logits_l3 * (1 - mask_l3))) * 1.5
        if mask_l4 is not None:
            consistency_loss += torch.mean(torch.relu(logits_l4 * (1 - mask_l4))) * 1.0
        return consistency_loss

    # ---------------- Forward ----------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_l1=None,
        labels_l2=None,
        labels_l3=None,
        labels_l4=None,
        **kwargs,
    ):
        # run base model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # (B, S, H)
        device = last_hidden.device
        B = last_hidden.size(0)

        # pooled hidden: last non-pad token if pad_token_id exists
        # ✅ 修复：检查pad_token_id是否为None
        if input_ids is not None and hasattr(self.config, "pad_token_id") and self.config.pad_token_id is not None:
            # 使用pad_token_id找到最后一个非padding token
            seq_lens = (input_ids != self.config.pad_token_id).sum(-1) - 1
            pooled = last_hidden[torch.arange(B, device=device), seq_lens]  # (B, H)
        elif attention_mask is not None:
            # 使用attention_mask找到最后一个有效token
            seq_lens = attention_mask.sum(-1) - 1  # (B,)
            pooled = last_hidden[torch.arange(B, device=device), seq_lens]  # (B, H)
        else:
            # 兜底：使用最后一个token
            pooled = last_hidden[:, -1, :]  # (B, H)
        
        # 🔬 药物知识融合（可选）
        drug_align_loss = None
        if self.drug_knowledge is not None:
            pooled, drug_align_loss = self.drug_knowledge(pooled, labels_l4)

        # 根据hierarchical_prediction决定计算哪些层的logits
        if self.hierarchical_prediction:
            # 层级模式：计算所有层的logits (L1->L2->L3->L4)
            # Level 1
            logits_l1 = self.cls_head_l1(pooled)

            # Level 2
            if self.use_cross_attention:
                ent2, ctx2, w2 = self._apply_cross_attention(
                    pooled, logits_l1, self.cls_head_l1.weight, attn_layer=self.cross_attn_l2, threshold=self.cross_threshold
                )
                h2 = self.norm_l2(pooled + self.cross_residual_scale * ctx2)
            else:
                h2 = pooled
                w2 = None  # 没有交叉注意力时设为None
            logits_l2 = self.cls_head_l2(h2)

            # Level 3
            if self.use_cross_attention:
                ent3, ctx3, w3 = self._apply_cross_attention(
                    h2, logits_l2, self.cls_head_l2.weight, attn_layer=self.cross_attn_l3, threshold=self.cross_threshold
                )
                h3 = self.norm_l3(h2 + self.cross_residual_scale * ctx3)
            else:
                h3 = h2
                w3 = None  # 没有交叉注意力时设为None
            logits_l3 = self.cls_head_l3(h3)

            # Level 4
            if self.use_cross_attention:
                ent4, ctx4, w4 = self._apply_cross_attention(
                    h3, logits_l3, self.cls_head_l3.weight, attn_layer=self.cross_attn_l4, threshold=self.cross_threshold
                )
                h4 = self.norm_l4(h3 + self.cross_residual_scale * ctx4)
            else:
                h4 = h3
                w4 = None  # 没有交叉注意力时设为None
            logits_l4 = self.cls_head_l4(h4)
        else:
            # 非层级模式：直接计算L4的logits，跳过L1-L3
            logits_l1 = None
            logits_l2 = None
            logits_l3 = None
            w2 = w3 = w4 = None
            ent2 = ent3 = ent4 = None
            # 直接使用pooled特征计算L4
            logits_l4 = self.cls_head_l4(pooled)

        # optionally print diagnostic info occasionally (only on rank 0 or single-GPU)
        self._forward_cnt += 1
        do_print = (self._forward_cnt % self.print_freq == 0)
        if do_print and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            try:
                if self.hierarchical_prediction and logits_l1 is not None:
                    p1 = torch.sigmoid(logits_l1)[0]
                    top1 = torch.topk(p1, k=min(5, p1.numel())).indices.tolist()
                    # 安全处理NaN值
                    ent2_val = round(float(ent2), 4) if ent2 is not None and not torch.isnan(torch.tensor(ent2)) else 'N/A'
                    ent3_val = round(float(ent3), 4) if ent3 is not None and not torch.isnan(torch.tensor(ent3)) else 'N/A'
                    ent4_val = round(float(ent4), 4) if ent4 is not None and not torch.isnan(torch.tensor(ent4)) else 'N/A'
                    print(f"🧠 [Cross-Attn] Sample0 L1 top idx: {top1}, entropies ~ [{ent2_val}, {ent3_val}, {ent4_val}]")
                elif not self.hierarchical_prediction:
                    p4 = torch.sigmoid(logits_l4)[0]
                    top4 = torch.topk(p4, k=min(5, p4.numel())).indices.tolist()
                    print(f"🧠 [Direct L4] Sample0 L4 top idx: {top4}")
            except Exception:
                pass

        # hierarchical masks and consistency loss
        if self.hierarchical_prediction:
            mask_l2, mask_l3, mask_l4 = self._generate_hierarchical_masks(logits_l1, logits_l2, logits_l3, logits_l4)
            consistency_loss = self._compute_consistency_loss(logits_l1, logits_l2, logits_l3, logits_l4, mask_l2, mask_l3, mask_l4)
        else:
            mask_l2 = mask_l3 = mask_l4 = None
            consistency_loss = 0.0

        # compute losses
        total_loss = None
        loss_terms = []
        if self.hierarchical_prediction:
            # 层级模式：计算所有层的损失
            if labels_l1 is not None and logits_l1 is not None:
                loss_l1 = F.binary_cross_entropy_with_logits(logits_l1, labels_l1.float())
                loss_terms.append(0.4 * loss_l1)
            if labels_l2 is not None and logits_l2 is not None:
                loss_l2 = F.binary_cross_entropy_with_logits(logits_l2, labels_l2.float())
                loss_terms.append(0.4 * loss_l2)
            if labels_l3 is not None and logits_l3 is not None:
                loss_l3 = F.binary_cross_entropy_with_logits(logits_l3, labels_l3.float())
                loss_terms.append(1.0 * loss_l3)
            if labels_l4 is not None:
                loss_l4 = F.binary_cross_entropy_with_logits(logits_l4, labels_l4.float())
                loss_terms.append(1.5 * loss_l4)
        else:
            # 非层级模式：只计算L4的损失
            if labels_l4 is not None:
                loss_l4 = F.binary_cross_entropy_with_logits(logits_l4, labels_l4.float())
                loss_terms.append(1.0 * loss_l4)

        if loss_terms:
            total_loss = sum(loss_terms)
            if self.hierarchical_prediction:
                total_loss = total_loss + 0.1 * consistency_loss
            if drug_align_loss is not None:
                total_loss = total_loss + 0.1 * drug_align_loss

        # 收集注意力权重（用于可视化）
        attention_weights = None
        if self.use_cross_attention:
            attention_weights = {
                'w2': w2,  # L2←L1 attention weights (B, N1)
                'w3': w3,  # L3←L2 attention weights (B, N2)
                'w4': w4,  # L4←L3 attention weights (B, N3)
            }

        # 根据hierarchical_prediction决定返回的logits格式
        if self.hierarchical_prediction:
            # 层级模式：返回所有层的logits (L1, L2, L3, L4)
            output_logits = (logits_l1, logits_l2, logits_l3, logits_l4)
        else:
            # 非层级模式：只返回L4的logits
            output_logits = logits_l4

        return SequenceClassifierOutputWithPast(
            loss=total_loss,
            logits=output_logits,
            hidden_states=pooled,
            attentions=attention_weights,  # 添加注意力权重
        )

    # ---------------- Helpers for code retrieval (kept from original) ----------------
    def _get_l1_code(self, idx):
        """获取L1代码"""
        if self.voc_l1 is not None and hasattr(self.voc_l1, "idx2word") and idx < len(self.voc_l1.idx2word):
            return self.voc_l1.idx2word[idx]
        else:
            return chr(ord('A') + idx % 14)

    def _get_l2_code(self, idx):
        if self.voc_l2 is not None and hasattr(self.voc_l2, "idx2word") and idx < len(self.voc_l2.idx2word):
            return self.voc_l2.idx2word[idx]
        else:
            return f"A{idx:02d}"

    def _get_l3_code(self, idx):
        if self.voc_l3 is not None and hasattr(self.voc_l3, "idx2word") and idx < len(self.voc_l3.idx2word):
            return self.voc_l3.idx2word[idx]
        else:
            return f"A{idx:02d}A"

    def _get_l4_code(self, idx):
        if self.voc_l4 is not None and hasattr(self.voc_l4, "idx2word") and idx < len(self.voc_l4.idx2word):
            return self.voc_l4.idx2word[idx]
        else:
            return f"A{idx:02d}AA"