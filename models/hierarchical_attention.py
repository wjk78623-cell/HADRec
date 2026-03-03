# 层级注意力校准模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HierarchicalAttentionCalibration(nn.Module):
    """
    层级注意力校准模块
    用于在层级分类中传递上层信息到下层，减少噪声传递
    """
    
    def __init__(self, config, upper_vocab_size, lower_vocab_size, hidden_size=None):
        super().__init__()
        
        self.upper_vocab_size = upper_vocab_size
        self.lower_vocab_size = lower_vocab_size
        self.hidden_size = hidden_size or config.hidden_size
        
        # 预学习的层级注意力矩阵 (upper_vocab, lower_vocab)
        # 表示每个上层类别对下层类别的注意力权重
        self.attention_weights = nn.Parameter(
            torch.randn(upper_vocab_size, lower_vocab_size) * 0.1
        )
        
        # 校准网络：融合原始特征和上层信息
        self.calibration_net = nn.Sequential(
            nn.Linear(self.hidden_size + upper_vocab_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # 动态阈值学习
        self.threshold_net = nn.Sequential(
            nn.Linear(upper_vocab_size, upper_vocab_size // 2),
            nn.ReLU(),
            nn.Linear(upper_vocab_size // 2, upper_vocab_size),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 注意力权重初始化为均匀分布
        nn.init.xavier_uniform_(self.attention_weights)
        
        # 校准网络权重初始化
        for module in self.calibration_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_features, upper_logits, upper_probs):
        """
        前向传播
        
        Args:
            hidden_features: 原始隐藏特征 (batch_size, hidden_size)
            upper_logits: 上层logits (batch_size, upper_vocab_size)
            upper_probs: 上层概率分布 (batch_size, upper_vocab_size)
        
        Returns:
            enhanced_features: 增强后的特征 (batch_size, hidden_size)
        """
        batch_size = hidden_features.shape[0]
        
        # 1. 计算上层注意力权重
        upper_attention = F.softmax(upper_probs, dim=1)  # (B, upper_vocab)
        
        # 2. 应用预学习的层级注意力矩阵
        # attention_weights: (upper_vocab, lower_vocab)
        hierarchical_attention = torch.matmul(
            upper_attention, self.attention_weights
        )  # (B, lower_vocab)
        
        # 3. 注意力加权的特征增强
        # 将注意力权重应用到特征维度
        attention_enhanced = hidden_features.unsqueeze(1) * hierarchical_attention.unsqueeze(2)
        attention_enhanced = attention_enhanced.mean(dim=1)  # (B, H)
        
        # 4. 特征融合
        combined = torch.cat([hidden_features, upper_probs], dim=1)
        calibrated = self.calibration_net(combined)
        
        # 5. 最终增强特征 = 原始特征 + 注意力增强 + 校准特征
        enhanced = hidden_features + 0.3 * attention_enhanced + 0.7 * calibrated
        
        return enhanced
    
    def compute_dynamic_threshold(self, upper_probs):
        """
        计算动态置信度阈值
        
        Args:
            upper_probs: 上层概率分布 (batch_size, upper_vocab_size)
        
        Returns:
            thresholds: 动态阈值 (batch_size, upper_vocab_size)
        """
        return self.threshold_net(upper_probs)
    
    def apply_threshold_filter(self, upper_probs, thresholds, min_threshold=0.5):
        """
        应用动态阈值过滤噪声
        
        Args:
            upper_probs: 上层概率分布 (batch_size, upper_vocab_size)
            thresholds: 动态阈值 (batch_size, upper_vocab_size)
            min_threshold: 最小阈值
        
        Returns:
            filtered_probs: 过滤后的概率分布
        """
        # 使用动态阈值和最小阈值的最大值
        effective_thresholds = torch.max(thresholds, 
                                       torch.full_like(thresholds, min_threshold))
        
        # 应用阈值过滤
        mask = upper_probs > effective_thresholds
        filtered_probs = upper_probs * mask.float()
        
        # 重新归一化
        filtered_probs = F.softmax(filtered_probs, dim=1)
        
        return filtered_probs


class HierarchicalConsistencyLoss(nn.Module):
    """
    层级一致性损失
    确保下层预测与上层预测保持一致
    """
    
    def __init__(self, upper_to_lower_mapping, alpha=0.1):
        """
        Args:
            upper_to_lower_mapping: 上层到下层类别的映射矩阵 (upper_vocab, lower_vocab)
            alpha: 损失权重
        """
        super().__init__()
        self.mapping = upper_to_lower_mapping
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, upper_probs, lower_probs):
        """
        计算层级一致性损失
        
        Args:
            upper_probs: 上层概率分布 (batch_size, upper_vocab_size)
            lower_probs: 下层概率分布 (batch_size, lower_vocab_size)
        
        Returns:
            consistency_loss: 一致性损失
        """
        # 将上层概率映射到下层空间
        mapped_upper = torch.matmul(upper_probs, self.mapping)  # (B, lower_vocab)
        
        # 计算KL散度损失
        upper_log = F.log_softmax(mapped_upper, dim=1)
        lower_soft = F.softmax(lower_probs, dim=1)
        
        consistency_loss = self.kl_loss(upper_log, lower_soft)
        
        return self.alpha * consistency_loss


class MultiLevelHierarchicalAttention(nn.Module):
    """
    多层级注意力机制
    支持多个层级之间的注意力传递
    """
    
    def __init__(self, config, vocab_sizes, hidden_size=None):
        """
        Args:
            config: 模型配置
            vocab_sizes: 各层级词汇表大小列表 [L1_size, L2_size, L3_size, L4_size]
            hidden_size: 隐藏层大小
        """
        super().__init__()
        
        self.num_levels = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        self.hidden_size = hidden_size or config.hidden_size
        
        # 创建层级间的注意力模块
        self.attention_modules = nn.ModuleList()
        for i in range(self.num_levels - 1):
            attention_module = HierarchicalAttentionCalibration(
                config, vocab_sizes[i], vocab_sizes[i + 1], self.hidden_size
            )
            self.attention_modules.append(attention_module)
        
        # 层级一致性损失模块
        self.consistency_losses = nn.ModuleList()
        for i in range(self.num_levels - 1):
            # 创建映射矩阵（这里需要根据实际的层级关系来定义）
            mapping = self._create_mapping_matrix(vocab_sizes[i], vocab_sizes[i + 1])
            consistency_loss = HierarchicalConsistencyLoss(mapping)
            self.consistency_losses.append(consistency_loss)
    
    def _create_mapping_matrix(self, upper_size, lower_size):
        """
        创建上层到下层类别的映射矩阵
        这里使用随机初始化，实际应用中应该根据真实的层级关系来定义
        """
        mapping = torch.randn(upper_size, lower_size)
        # 归一化每行，使其和为1
        mapping = F.softmax(mapping, dim=1)
        return mapping
    
    def forward(self, hidden_features, level_logits, level_probs):
        """
        多层级前向传播
        
        Args:
            hidden_features: 原始隐藏特征 (batch_size, hidden_size)
            level_logits: 各层级logits列表
            level_probs: 各层级概率分布列表
        
        Returns:
            enhanced_features_list: 各层级增强后的特征列表
        """
        enhanced_features = [hidden_features]
        
        for i in range(self.num_levels - 1):
            # 使用当前层级的特征和上层信息进行增强
            enhanced = self.attention_modules[i](
                enhanced_features[-1], level_logits[i], level_probs[i]
            )
            enhanced_features.append(enhanced)
        
        return enhanced_features
    
    def compute_consistency_loss(self, level_probs):
        """
        计算层级一致性损失
        
        Args:
            level_probs: 各层级概率分布列表
        
        Returns:
            total_consistency_loss: 总的一致性损失
        """
        total_loss = 0.0
        
        for i in range(self.num_levels - 1):
            consistency_loss = self.consistency_losses[i](
                level_probs[i], level_probs[i + 1]
            )
            total_loss += consistency_loss
        
        return total_loss



