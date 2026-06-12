import torch
import torch.nn as nn
import torch.nn.functional as F

class CellFILIPLoss(nn.Module):
    def __init__(self, temperature=0.07, att_feat_path='text_features/biomedclip/biomedclip_att6_embeddings.pth'):
        super(CellFILIPLoss, self).__init__()
        self.temperature = temperature
        self.text_features = torch.load(att_feat_path, map_location='cpu').cuda()
        # self.text_features = F.normalize(self.text_features, dim=-1)    # (num_class, N_T, d)

    def forward(self, image_features, labels):
        """
        image_features: (B, N_I, d) - 图像块的特征
        text_features: (num_class, N_T, d) - 每个类别的文本token特征
        labels: (B,) - 每张图像的类别标签
        """

        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)  # (B, N_I, d)

        # 计算图像与每个类别的最大相似度
        image_to_text_sim = torch.einsum('bid,cjd->bcij', image_features, self.text_features)  # (B, num_class, N_I, N_T)
        image_to_text_max_sim = image_to_text_sim.max(dim=-1).values  # (B, num_class, N_I)
        S_I = image_to_text_max_sim.mean(dim=-1)  # (B, num_class)

        # 计算每张图像与其对应类别的匹配分数
        logits = S_I / self.temperature  # (B, num_class)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss


# 示例用法
B, N_I, N_T, d = 32, 196, 77, 256  # batch size, 图像块数, 文本token数, 嵌入维度
num_class = 10  # 类别数

image_features = torch.randn(B, N_I, d)  # 随机生成图像特征
text_features = torch.randn(num_class, N_T, d)  # 随机生成文本特征
labels = torch.randint(0, num_class, (B,))  # 随机生成类别标签

modified_filip_loss = CellFILIPLoss(temperature=0.07)
loss = modified_filip_loss(image_features, text_features, labels)
print("Modified FILIP Loss:", loss.item())