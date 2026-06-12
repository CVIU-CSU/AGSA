import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS

@MODELS.register_module()
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, weight=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.weight = weight
        
    def forward(self, features, **kwargs):
        """
        手动计算InfoNCE损失
        
        Args:
            features: 输入特征 [batch_size, feature_dim]
        """
        batch_size = features.shape[0]
        
        # 特征归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建正样本掩码（对角线）
        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        
        # 提取正样本相似度
        pos_sim = sim_matrix[pos_mask].view(batch_size, -1)
        
        # 提取负样本相似度
        neg_sim = sim_matrix[~pos_mask].view(batch_size, batch_size - 1)
        
        # 手动计算softmax和交叉熵
        # 对于每个样本，正样本相似度在第一个位置
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        
        loss = self.weight * F.cross_entropy(logits, labels)

        return loss