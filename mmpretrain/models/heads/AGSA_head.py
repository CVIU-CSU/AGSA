# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cls_head import ClsHead


from typing import List
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.structures import DataSample

import numpy as np
import math


from typing import Callable
from torch import Tensor

import json
import os


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MFA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=5):
        super().__init__()
        self.num_heads = num_heads

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        """
        x: (B, 512)
        return: list of 5 tensors, each shape (B, 512)
        """
        return [head(x) for head in self.heads]

@MODELS.register_module()
class AGSAClsHead(ClsHead):
    """

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: tuple,
                 loss_aux=None,
                 loss_supcon=None,
                 area_mean=None,
                 area_std=None,

                 all_img_path = 'xxx/all_img_embeddings.pt',
                 mapper_weight_path = 'xxx/MFA.pth',
                 root_prefix_path = 'xxx/',

                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(AGSAClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        # load
        self.all_img_embeddings = torch.load(all_img_path)
        for k, v in self.all_img_embeddings.items():
            self.all_img_embeddings[k] = self.all_img_embeddings[k].cuda()
            # self.all_img_embeddings[k] = self.all_img_embeddings[k].cpu() # save GPU memory

        self.biomedclip_att_head = MFA(embed_dim=512, num_heads=5)
        self.biomedclip_att_head.load_state_dict(torch.load(mapper_weight_path, map_location='cpu'))
        # freeze
        for param in self.biomedclip_att_head.parameters():
            param.requires_grad = False

        self.root_prefix_path = root_prefix_path
        self.prefix_len = len(self.root_prefix_path)

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        if loss_aux and not isinstance(loss_aux, nn.Module):
            self.loss_aux = MODELS.build(loss_aux)
        if loss_supcon and not isinstance(loss_supcon, nn.Module):
            self.loss_supcon = MODELS.build(loss_supcon)

        self.area_mean = area_mean
        self.area_std = area_std

        self.text_feat_dim = 512
        self.weight_predictor = Mlp(self.text_feat_dim * 6, self.text_feat_dim * 2, 6, drop=0.4)
        # self.weight_predictor = Mlp(self.text_feat_dim * 6, self.text_feat_dim * 2, 6, drop=0.1)
        self.size_mapper = Mlp(self.text_feat_dim, self.text_feat_dim, self.text_feat_dim)
        self.fc = Mlp(self.in_channels + self.text_feat_dim, self.text_feat_dim, self.num_classes)
        self.fc_aux = Mlp(self.text_feat_dim, self.text_feat_dim, self.num_classes)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        final_feat = feats[-1]

        return final_feat

    def batch_sinusoidal_encoding(self, x):
        """
        对批量数据生成正余弦编码
        
        参数:
            x: 输入Tensor，形状为 (batch_size,) 或 (...,)
            d: 目标维度（输出特征维度）
        
        返回:
            形状为 (..., d) 的编码Tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # 预计算分母项
        device = x.device
        position = torch.arange(0, self.text_feat_dim, 2, dtype=torch.float32, device=device)
        div_term = torch.exp(position * (-math.log(10000.0) / self.text_feat_dim))  # 10000^(-2i/d)
        
        # 扩展输入维度
        x_expanded = x.unsqueeze(-1)  # (..., 1)
        
        # 初始化输出Tensor
        pe = torch.zeros(*x.shape, self.text_feat_dim, device=device)
        
        # 向量化计算
        pe[..., 0::2] = torch.sin(x_expanded * div_term)  # 偶数位置sin
        pe[..., 1::2] = torch.cos(x_expanded * div_term)  # 奇数位置cos

        return pe

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx

        final_feat = self(feats)

        areas = []
        all_att_feat1 = []
        for i in data_samples:
            areas.append((i.ori_shape[0]*i.ori_shape[1] - self.area_mean) / self.area_std)
            # all_att_feat1.append(self.all_img_embeddings[i.img_path[self.prefix_len:]].cuda())
            now_k = i.img_path[self.prefix_len:]
            assert now_k in self.all_img_embeddings.keys()
            all_att_feat1.append(self.all_img_embeddings[now_k])
        all_att_feat1 = torch.stack(all_att_feat1, dim=0)
        # all_att_feat1 = all_att_feat1.cuda() # save GPU memory
        areas = torch.tensor(areas).cuda()
        areas_encoding = self.batch_sinusoidal_encoding(areas)
        size_feat = self.size_mapper(areas_encoding).unsqueeze(1)

        all_att_feat = self.biomedclip_att_head(all_att_feat1)
        all_att_feat = torch.cat(all_att_feat, dim=1)

        all_feats = torch.cat([all_att_feat, size_feat], dim=1)
        concat2_feat = all_feats.reshape(all_feats.shape[0], -1) # B*(6*512)
        weight = self.weight_predictor(concat2_feat) # B*6
        # weight = torch.softmax(weight, dim=1)
        weight = torch.softmax(weight/3.0, dim=1)
        weight = weight.unsqueeze(-1) # B*6*1
        union_feat = weight * all_feats
        union_feat = union_feat.sum(dim=1) # B*512
        cls_score = self.fc(torch.cat([union_feat, final_feat], dim=-1))
        cls_score_aux = self.fc_aux(union_feat)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, cls_score_aux, union_feat, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor, cls_score_aux, union_feat,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss_cls'] = loss

        loss_cls_aux = self.loss_aux(
            cls_score_aux, target, avg_factor=cls_score_aux.size(0), **kwargs)
        losses['loss_cls_aux'] = loss_cls_aux

        loss_supcon = self.loss_supcon(
            union_feat, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss_supcon'] = loss_supcon

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """

        # The part can be traced by torch.fx
        final_feat = self(feats)

        areas = []
        all_att_feat1 = []
        for i in data_samples:
            areas.append((i.ori_shape[0]*i.ori_shape[1] - self.area_mean) / self.area_std)
            # all_att_feat1.append(self.all_img_embeddings[i.img_path[self.prefix_len:]].cuda())
            now_k = i.img_path[self.prefix_len:]
            assert now_k in self.all_img_embeddings.keys()
            all_att_feat1.append(self.all_img_embeddings[now_k])
        all_att_feat1 = torch.stack(all_att_feat1, dim=0)
        # all_att_feat1 = all_att_feat1.cuda() # save GPU memory
        areas = torch.tensor(areas).cuda()
        areas_encoding = self.batch_sinusoidal_encoding(areas)
        size_feat = self.size_mapper(areas_encoding).unsqueeze(1)

        all_att_feat = self.biomedclip_att_head(all_att_feat1)
        all_att_feat = torch.cat(all_att_feat, dim=1)

        all_feats = torch.cat([all_att_feat, size_feat], dim=1)
        concat2_feat = all_feats.reshape(all_feats.shape[0], -1) # B*(6*512)
        weight = self.weight_predictor(concat2_feat) # B*6
        # weight = torch.softmax(weight, dim=1)
        weight = torch.softmax(weight/3.0, dim=1)
        weight = weight.unsqueeze(-1) # B*6*1
        union_feat = weight * all_feats
        union_feat = union_feat.sum(dim=1) # B*512
        cls_score = self.fc(torch.cat([union_feat, final_feat], dim=-1))
        # cls_score_aux = self.fc_aux(union_feat)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)

        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples