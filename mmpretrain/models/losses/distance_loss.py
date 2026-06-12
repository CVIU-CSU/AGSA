import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS

@MODELS.register_module()
class DistanceLoss(nn.Module):
    def __init__(self, loss_weight=1.0,):
        super(DistanceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, y, **kwargs):
        # 确保y不需要梯度更新
        y = y.detach()
        
        # 计算x和y之间的欧几里得距离
        distances = F.pairwise_distance(x, y, p=2)
        
        # 计算平均距离损失
        loss_dis = distances.mean()

        loss = self.loss_weight * loss_dis

        return loss

if __name__ == '__main__':
    # loss = DistanceLoss(loss_weight=0.15)
    loss = DistanceLoss()
    x = torch.randn(128, 512)
    y = torch.randn(128, 512)
    print(loss(x, y))