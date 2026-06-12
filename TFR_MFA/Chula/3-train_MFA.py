import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# 超参
BATCH_SIZE = 16
NUM_EPOCHS = 20
DATA_PATH = 'xxx/all_img_embeddings.pt'
TEXT_PATH = 'xxx/Refined_Attribute_Text_Feature.pth'
EMBED_DIM = 512
TEMPERATURE = 0.5
LR = 1e-4
MODEL_SAVE_PATH = 'xxx/MFA.pth'
device = "cuda:0"
atts_list = ['shape', 'nucleus', 'chromatin', 'cytoplasm', 'ratio']

CELL_CATEGORIES = ('0', '1', '10', '11', '12', '2', '3', '4', '5', '6', '7', '8', '9')

class FeatureDataset(Dataset):
    def __init__(self, features_dict, split="train"):
        self.data = []
        self.label = []

        for key, feat in features_dict.items():
            tmp_list = key.split('/')
            assert len(tmp_list) == 3
            data_type, cls_name, img_name = tmp_list
            if 'neg' in cls_name:
                continue
            if data_type == split:
                self.data.append(feat)
                self.label.append(CELL_CATEGORIES.index(cls_name))

        self.data = torch.stack(self.data)  # (N, 512)
        self.label = torch.tensor(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]    # 返回 (512,)

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

def train_one_epoch(model, train_loader, optimizer, text_features):
    model.train()
    for batch, (img_feat, label) in enumerate(train_loader):
        img_feat, label = img_feat.to(device), label.to(device)
        img_feat = img_feat.squeeze(1)    # (B,512)

        outputs = model(img_feat)          # list of 5 tensors (B,512)
        
        loss_dic = dict()
        for i in range(len(atts_list)):
            pred_logit = F.normalize(outputs[i], dim=-1, p=2) @ F.normalize(text_features[atts_list[i] + '_des_pool'], dim=-1, p=2) / TEMPERATURE
            att_gt = text_features[atts_list[i] + '_cls_label'][label]
            # 计算损失
            loss_dic['loss_' + atts_list[i]] = F.cross_entropy(pred_logit, att_gt)

        loss = sum(loss_dic.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            # 打印每个属性头的损失
            print({k: v.item() for k, v in loss_dic.items()})

def val_one_epoch(model, val_loader, text_features):
    model.eval()
    with torch.no_grad():
        all_pred = [[], [], [], [], []]
        all_att_gt = [[], [], [], [], []]
        for img_feat, label in val_loader:
            img_feat, label = img_feat.to(device), label.to(device)
            img_feat = img_feat.squeeze(1)

            outputs = model(img_feat)

            for i in range(len(atts_list)):
                pred_logit = F.normalize(outputs[i], dim=-1, p=2) @ F.normalize(text_features[atts_list[i] + '_des_pool'], dim=-1, p=2) / TEMPERATURE
                pred = F.softmax(pred_logit, dim=-1)
                pred_cls = torch.argmax(pred, dim=-1)
                att_gt = text_features[atts_list[i] + '_cls_label'][label]

                all_pred[i].append(pred_cls)
                all_att_gt[i].append(att_gt)

    # 计算每个属性头的准确率
    all_acc = 0
    for i in range(len(atts_list)):
        pred = torch.cat(all_pred[i], dim=0)
        att_gt = torch.cat(all_att_gt[i], dim=0)
        acc = (pred == att_gt).float().mean()
        print(f'{atts_list[i]} acc: {acc.item()}')

        all_acc += acc
    print(f'Average acc: {all_acc / len(atts_list)}')
    
    return all_acc / len(atts_list)

if __name__ == "__main__":
    all_data = torch.load(DATA_PATH)

    train_dataset = FeatureDataset(all_data, split="train")
    val_dataset   = FeatureDataset(all_data, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MFA(embed_dim=EMBED_DIM, num_heads=len(atts_list)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    text_features = torch.load(TEXT_PATH)
    for att in atts_list:
        text_features[att + '_des_pool'] = text_features[att + '_des_pool'].T.to(device)
        text_features[att + '_cls_label'] = text_features[att + '_cls_label'].to(device)

    best_acc = -1
    best_epoch = -1
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, text_features)
        acc = val_one_epoch(model, val_loader, text_features)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f'Best acc: {best_acc} at epoch {best_epoch}')