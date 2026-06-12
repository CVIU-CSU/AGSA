import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS

@MODELS.register_module()
class SimilarityLoss(nn.Module):
    def __init__(self, loss_weight=1.0,):
        super(SimilarityLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, y, **kwargs):
        # 计算x和y的相似性矩阵
        sim_matrix_x = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        sim_matrix_y = F.cosine_similarity(y.unsqueeze(1), y.unsqueeze(0), dim=2)
        
        # 确保y的相似性矩阵不需要梯度
        sim_matrix_y = sim_matrix_y.detach()
        
        # 计算相似度损失，这里使用均方误差
        loss_sim = F.mse_loss(sim_matrix_x, sim_matrix_y, reduction='sum')

        loss = self.loss_weight * loss_sim

        return loss