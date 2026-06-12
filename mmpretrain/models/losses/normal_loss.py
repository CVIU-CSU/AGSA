import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS

@MODELS.register_module()
class NormalLoss(nn.Module):
    def __init__(self, loss_weight=1.0,):
        super(NormalLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, mul, std, **kwargs):
        variance_dul = std ** 2
        variance_dul = variance_dul.view(variance_dul.shape[0], -1)
        mul = mul.view(mul.shape[0], -1)

        loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
        loss_kl = torch.mean(loss_kl)

        loss = self.loss_weight * loss_kl

        return loss