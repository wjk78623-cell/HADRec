# 性能提升方法的具体实现
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DynamicThresholdFilter(nn.Module):
    """
    动态阈值过滤模块
    根据验证集性能动态调整每个类别的置信度阈值
    """
    
    def __init__(self, num_classes, initial_threshold=0.5, learning_rate=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # 可学习的阈值参数
        self.thresholds = nn.Parameter(torch.full((num_classes,), initial_threshold))
        
        # 阈值更新统计
        self.register_buffer('update_count', torch.zeros(num_classes))
        self.register_buffer('performance_history', torch.zeros(num_classes, 10))  # 保存最近10次的性能
        
    def forward(self, logits, labels=None, update_thresholds=False):
        """
        前向传播
        
        Args:
            logits: 模型输出logits (batch_size, num_classes)
            labels: 真实标签 (batch_size, num_classes) - 仅在训练时使用
            update_thresholds: 是否更新阈值
        
        Returns:
            filtered_probs: 过滤后的概率分布
        """
        probs = torch.sigmoid(logits)
        
        if self.training and update_thresholds and labels is not None:
            self._update_thresholds(probs, labels)
        
        # 应用动态阈值过滤
        filtered_probs = self._apply_threshold_filter(probs)
        
        return filtered_probs
    
    def _apply_threshold_filter(self, probs):
        """应用阈值过滤"""
        # 使用sigmoid确保阈值在(0,1)范围内
        effective_thresholds = torch.sigmoid(self.thresholds)
        
        # 创建掩码：只保留超过阈值的预测
        mask = probs > effective_thresholds.unsqueeze(0)
        
        # 应用掩码
        filtered_probs = probs * mask.float()
        
        # 重新归一化
        filtered_probs = F.softmax(filtered_probs, dim=1)
        
        return filtered_probs
    
    def _update_thresholds(self, probs, labels):
        """根据性能更新阈值"""
        with torch.no_grad():
            # 计算每个类别的F1分数
            for class_idx in range(self.num_classes):
                pred_class = (probs[:, class_idx] > torch.sigmoid(self.thresholds[class_idx])).float()
                true_class = labels[:, class_idx]
                
                # 计算F1分数
                tp = (pred_class * true_class).sum()
                fp = (pred_class * (1 - true_class)).sum()
                fn = ((1 - pred_class) * true_class).sum()
                
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0.0
                
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0.0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                # 更新性能历史
                self.performance_history[class_idx, self.update_count[class_idx] % 10] = f1
                self.update_count[class_idx] += 1
                
                # 如果性能下降，调整阈值
                if self.update_count[class_idx] > 1:
                    recent_performance = self.performance_history[class_idx, :5].mean()
                    older_performance = self.performance_history[class_idx, 5:].mean()
                    
                    if recent_performance < older_performance:
                        # 性能下降，降低阈值
                        self.thresholds[class_idx] -= self.learning_rate
                    else:
                        # 性能提升，提高阈值
                        self.thresholds[class_idx] += self.learning_rate


class HardConstraintPropagation(nn.Module):
    """
    硬约束传递模块
    将上层预测结果作为底层输入特征，强制底层模型关注上层已确定的类别方向
    """
    
    def __init__(self, upper_vocab_size, lower_vocab_size, hidden_size):
        super().__init__()
        self.upper_vocab_size = upper_vocab_size
        self.lower_vocab_size = lower_vocab_size
        self.hidden_size = hidden_size
        
        # 上层到下层类别的映射矩阵
        self.mapping_matrix = nn.Parameter(
            torch.randn(upper_vocab_size, lower_vocab_size) * 0.1
        )
        
        # 特征融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size + upper_vocab_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 约束强度控制
        self.constraint_strength = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, hidden_features, upper_probs):
        """
        前向传播
        
        Args:
            hidden_features: 原始隐藏特征 (batch_size, hidden_size)
            upper_probs: 上层概率分布 (batch_size, upper_vocab_size)
        
        Returns:
            enhanced_features: 增强后的特征
            constraint_loss: 约束损失
        """
        # 1. 将上层概率映射到下层空间
        mapped_upper = torch.matmul(upper_probs, self.mapping_matrix)  # (B, lower_vocab)
        
        # 2. 特征融合
        combined_features = torch.cat([hidden_features, upper_probs], dim=1)
        enhanced_features = self.fusion_net(combined_features)
        
        # 3. 计算约束损失
        constraint_loss = self._compute_constraint_loss(hidden_features, mapped_upper)
        
        return enhanced_features, constraint_loss
    
    def _compute_constraint_loss(self, hidden_features, mapped_upper):
        """计算约束损失"""
        # 使用余弦相似度确保特征与映射的上层信息一致
        hidden_norm = F.normalize(hidden_features, p=2, dim=1)
        mapped_norm = F.normalize(mapped_upper, p=2, dim=1)
        
        # 计算余弦相似度损失
        cosine_sim = torch.sum(hidden_norm * mapped_norm, dim=1)
        constraint_loss = 1 - cosine_sim.mean()  # 1 - 余弦相似度
        
        return self.constraint_strength * constraint_loss


class SoftConstraintRegularization(nn.Module):
    """
    软约束正则化模块
    在底层损失函数中加入上层预测的对齐项，惩罚与上层逻辑冲突的预测
    """
    
    def __init__(self, upper_vocab_size, lower_vocab_size, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
        # 底层药品到上层类别的映射矩阵
        self.lower_to_upper_mapping = nn.Parameter(
            torch.randn(lower_vocab_size, upper_vocab_size) * 0.1
        )
        
        # KL散度损失
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, upper_probs, lower_probs):
        """
        计算软约束正则化损失
        
        Args:
            upper_probs: 上层概率分布 (batch_size, upper_vocab_size)
            lower_probs: 下层概率分布 (batch_size, lower_vocab_size)
        
        Returns:
            regularization_loss: 正则化损失
        """
        # 将下层概率映射到上层空间
        mapped_lower = torch.matmul(lower_probs, self.lower_to_upper_mapping)  # (B, upper_vocab)
        
        # 计算KL散度损失
        upper_log = F.log_softmax(upper_probs, dim=1)
        lower_soft = F.softmax(mapped_lower, dim=1)
        
        kl_loss = self.kl_loss(upper_log, lower_soft)
        
        return self.alpha * kl_loss


class AdaptiveNoiseFilter(nn.Module):
    """
    自适应噪声过滤模块
    根据预测置信度动态过滤噪声预测
    """
    
    def __init__(self, num_classes, confidence_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # 可学习的置信度阈值
        self.adaptive_thresholds = nn.Parameter(
            torch.full((num_classes,), confidence_threshold)
        )
        
        # 噪声检测网络
        self.noise_detector = nn.Sequential(
            nn.Linear(num_classes, num_classes // 2),
            nn.ReLU(),
            nn.Linear(num_classes // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, logits, return_noise_mask=False):
        """
        前向传播
        
        Args:
            logits: 模型输出logits (batch_size, num_classes)
            return_noise_mask: 是否返回噪声掩码
        
        Returns:
            filtered_logits: 过滤后的logits
            noise_mask: 噪声掩码 (可选)
        """
        probs = torch.sigmoid(logits)
        
        # 检测噪声预测
        noise_scores = self.noise_detector(probs)
        
        # 计算自适应阈值
        adaptive_thresholds = torch.sigmoid(self.adaptive_thresholds)
        
        # 创建噪声掩码
        noise_mask = (noise_scores < adaptive_thresholds.unsqueeze(0)).float()
        
        # 应用噪声过滤
        filtered_logits = logits * noise_mask
        
        if return_noise_mask:
            return filtered_logits, noise_mask
        else:
            return filtered_logits


class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合模块
    融合不同层级的特征表示
    """
    
    def __init__(self, feature_dims, output_dim):
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # 特征投影层
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # 注意力权重学习
        self.attention_weights = nn.Sequential(
            nn.Linear(output_dim * len(feature_dims), len(feature_dims)),
            nn.Softmax(dim=1)
        )
        
        # 最终融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, features_list):
        """
        前向传播
        
        Args:
            features_list: 特征列表，每个元素的形状为 (batch_size, feature_dim)
        
        Returns:
            fused_features: 融合后的特征 (batch_size, output_dim)
        """
        # 投影到统一维度
        projected_features = []
        for i, features in enumerate(features_list):
            projected = self.projections[i](features)
            projected_features.append(projected)
        
        # 拼接所有投影特征
        concatenated = torch.cat(projected_features, dim=1)
        
        # 计算注意力权重
        attention_weights = self.attention_weights(concatenated)  # (B, num_features)
        
        # 加权融合
        weighted_features = torch.zeros_like(projected_features[0])
        for i, features in enumerate(projected_features):
            weighted_features += attention_weights[:, i:i+1] * features
        
        # 最终融合
        fused_features = self.fusion_layer(weighted_features)
        
        return fused_features


class PerformanceEnhancementSuite(nn.Module):
    """
    性能提升套件
    整合所有性能提升方法
    """
    
    def __init__(self, config, vocab_sizes, hidden_size):
        super().__init__()
        
        self.num_levels = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        self.hidden_size = hidden_size
        
        # 动态阈值过滤
        self.dynamic_thresholds = nn.ModuleList([
            DynamicThresholdFilter(vocab_size) for vocab_size in vocab_sizes
        ])
        
        # 硬约束传递
        self.hard_constraints = nn.ModuleList()
        for i in range(self.num_levels - 1):
            constraint = HardConstraintPropagation(
                vocab_sizes[i], vocab_sizes[i + 1], hidden_size
            )
            self.hard_constraints.append(constraint)
        
        # 软约束正则化
        self.soft_constraints = nn.ModuleList()
        for i in range(self.num_levels - 1):
            constraint = SoftConstraintRegularization(
                vocab_sizes[i], vocab_sizes[i + 1]
            )
            self.soft_constraints.append(constraint)
        
        # 自适应噪声过滤
        self.noise_filters = nn.ModuleList([
            AdaptiveNoiseFilter(vocab_size) for vocab_size in vocab_sizes
        ])
        
        # 多尺度特征融合
        self.feature_fusion = MultiScaleFeatureFusion(
            [hidden_size] * self.num_levels, hidden_size
        )
        
    def forward(self, hidden_features, level_logits, level_probs, labels=None):
        """
        前向传播
        
        Args:
            hidden_features: 原始隐藏特征
            level_logits: 各层级logits列表
            level_probs: 各层级概率分布列表
            labels: 真实标签 (可选)
        
        Returns:
            enhanced_logits: 增强后的logits列表
            total_loss: 总损失
        """
        enhanced_logits = []
        total_loss = 0.0
        
        for i, (logits, probs) in enumerate(zip(level_logits, level_probs)):
            # 1. 动态阈值过滤
            filtered_probs = self.dynamic_thresholds[i](
                logits, labels[:, i] if labels is not None else None
            )
            
            # 2. 自适应噪声过滤
            filtered_logits = self.noise_filters[i](logits)
            
            # 3. 硬约束传递 (除了第一层)
            if i > 0:
                enhanced_features, constraint_loss = self.hard_constraints[i-1](
                    hidden_features, level_probs[i-1]
                )
                total_loss += constraint_loss
                
                # 使用增强特征重新计算logits
                filtered_logits = filtered_logits + enhanced_features.mean(dim=1, keepdim=True)
            
            enhanced_logits.append(filtered_logits)
        
        # 4. 软约束正则化
        for i in range(self.num_levels - 1):
            soft_loss = self.soft_constraints[i](
                level_probs[i], level_probs[i + 1]
            )
            total_loss += soft_loss
        
        return enhanced_logits, total_loss



