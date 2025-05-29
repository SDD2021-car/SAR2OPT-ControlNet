import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open_clip
import os
import json
from tqdm import tqdm

class ImageTextJSONDataset(Dataset):
    def __init__(self, json_path, preprocess):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.entries = json.load(f)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_path = self.entries[idx]["image"]  # 或 "trainB" 取决于你 json 字段名
        prompt = self.entries[idx]["prompt"]
        image = self.preprocess(Image.open(image_path).convert("RGB"))
        return image, prompt

# # 1. 定义你的图文数据集
# class ImageTextDataset(Dataset):
#     def __init__(self, image_dir, image_files, text_prompts, preprocess):
#         self.image_dir = image_dir
#         self.image_files = image_files
#         self.text_prompts = text_prompts
#         self.preprocess = preprocess
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_files[idx])
#         image = self.preprocess(Image.open(image_path).convert("RGB"))
#         text = self.text_prompts[idx]
#         return image, text

# 2. 准备图像路径和文本
# 3. 加载CLIP模型
device = torch.device("cuda:3")

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='/data/yjy_data/B2DiffRL/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin')
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model = model.to(device)
model.train()
# 2. 加载 JSON 数据集
json_path = "/data/yjy_data/B2DiffRL/FineTuneClip/SEN_season_JSON.json"
dataset = ImageTextJSONDataset(json_path, preprocess)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 5. 设置优化器和损失
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

n_epoch = 100
# 6. 开始训练
for epoch in range(10):
    print(f"\nEpoch {epoch+1}")
    epoch_loss = 0
    for images, prompts in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        images = images.to(device)
        prompts = tokenizer(prompts).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(prompts)

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T  # 相似度矩阵
        labels = torch.arange(len(images), device=images.device)  # 正确匹配为对角线

        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
save_path = "/data/yjy_data/B2DiffRL/FineTuneClip/clip_finetuned_season.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)


