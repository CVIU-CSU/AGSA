import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TFR(nn.Module):
    def __init__(self, dim=512, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.linear = nn.Linear(dim, out_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


def cosine_similarity_matrix(x):
    # x: (N, D)
    x_norm = F.normalize(x, dim=1)
    return x_norm @ x_norm.T  # (N, N)


# 超参数
alpha = 0.6     # 相似度比例
lambda_sim = 1.0  # 相似度损失的权重
lr = 1e-4
all_steps = 1000
out_dim = 576

att_pool = torch.load('xxx/att_text_raw_feat.pth')
dim = 512

cols = ['ratio', 'nucleus', 'shape', 'chromatin', 'cytoplasm']

model = TFR(dim)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for step in range(all_steps):
    for col in cols:
        x = att_pool[col + '_des_pool']

        y = model(x)
        
        # --- 损失 1: 重建 ---
        loss_mse = F.mse_loss(y, x)
        
        # --- 损失 2: 相似度矩阵保持 ---
        s_x = cosine_similarity_matrix(x)
        s_y = cosine_similarity_matrix(y)

        B = s_x.size(0)
        device = s_x.device

        # 对角线 mask
        diag_mask = torch.eye(B, device=device).bool()
        off_diag_mask = ~diag_mask

        target = s_x.clone()
        target[off_diag_mask] = alpha * s_x[off_diag_mask]
        loss_sim = F.mse_loss(s_y, target)
        
        # --- 总损失 ---
        loss = loss_mse + lambda_sim * loss_sim
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, mse={loss_mse.item():.4f}, sim={loss_sim.item():.4f}")


print('-' * 50)
# 查看其相互的余弦相似度矩阵
new_5att_pool = dict()
for col in cols:
    print(col)
    x = att_pool[col + '_des_pool']
    y = model(x)

    new_5att_pool[col + '_des_pool'] = y.detach().cpu()
    new_5att_pool[col + '_cls_label'] = att_pool[col + '_cls_label']

    print('before')
    cos = cosine_similarity_matrix(x)
    # print(cos)
    print(cos.mean())

    print('after')
    y = model(x)
    cos = cosine_similarity_matrix(y)
    # print(cos)
    print(cos.mean())

    # 查看y和x的相似度
    # print(F.cosine_similarity(y, x, dim=1))
    print('x and y similarity: ', F.cosine_similarity(y, x, dim=1).mean())

# 存储训练后的模型和数据
torch.save(model.state_dict(), 'xxx/TFR.pth')

torch.save(new_5att_pool, f'xxx/Refined_Attribute_Text_Feature.pth')


# 测试两个属性描述池之间的相似性
print('-' * 50)
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        col1 = cols[i]
        col2 = cols[j]
        pool1 = f'{col1}_des_pool'
        pool2 = f'{col2}_des_pool'
        embeddings1 = new_5att_pool[pool1]
        embeddings2 = new_5att_pool[pool2]
        embeddings1 = F.normalize(embeddings1, dim=-1, p=2)
        embeddings2 = F.normalize(embeddings2, dim=-1, p=2)
        similarity = torch.matmul(embeddings1,embeddings2.T)
        similarity_np = similarity.cpu().detach().numpy()
        print(f'pool {pool1} and {pool2} similarity')
        # print(similarity_np)
        print(similarity_np.mean())