from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import open_clip
import os
import torch
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
random.seed(3407)
from PIL import Image
from pathlib import Path

root_img_path = 'xxx/data/BCFC-329k/'
data_type = ['train', 'val', 'test']
context_length = 256
BS = 64
device = torch.device('cuda:4') if torch.cuda.is_available() else torch.device('cpu')
save_path = 'xxx/all_img_embeddings.pt'

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.to(device)
model.eval()

print('load model over')

all_class_embeddings = dict() # 存储全部数据

debug_cnt = BS * 1000
with torch.no_grad():
    all_paths = []
    for data_type in data_type:
        for data_path in os.listdir(os.path.join(root_img_path, data_type)):
            all_paths.extend(list(Path(os.path.join(root_img_path, data_type, data_path)).glob('*.jpg')))
    for start in range(0, len(all_paths), BS):
        if start % (BS * 100) == 0:
            print(f"Processing of the {start//BS}-th batch.")
        bs_paths = all_paths[start:start+BS]
        imgs_tensor = torch.stack([preprocess_val(Image.open(img_path)).to(device) for img_path in bs_paths])
        class_embeddings = model.encode_image(imgs_tensor)
        for i, img_path in enumerate(bs_paths):
            all_class_embeddings[img_path._str[len(root_img_path):]] = class_embeddings[i].unsqueeze(0).detach().cpu()

        if start % debug_cnt == 0:
            torch.save(all_class_embeddings, save_path)

# 最后保存一次
torch.save(all_class_embeddings, save_path)