import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
import os
import json
from tqdm import tqdm

# 1. 数据集类
class ImageTextJSONDataset(Dataset):
    def __init__(self, json_path, preprocess):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.entries = json.load(f)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_path = self.entries[idx]["image"]  # 图像路径字段
        prompt = self.entries[idx]["prompt"]     # 文本提示字段
        image = self.preprocess(Image.open(image_path).convert("RGB"))
        return image, prompt, image_path

# # 2. 模型准备
# device = torch.device("cuda:5")
# model, _, preprocess = open_clip.create_model_and_transforms(
#     'ViT-H-14',
#     pretrained='/data/yjy_data/B2DiffRL/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin'
# )
# tokenizer = open_clip.get_tokenizer('ViT-H-14')
# model = model.to(device)
# model.eval()  # 设置为评估模式！
device = torch.device("cuda:3")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14',
    pretrained=None  # 不加载默认预训练权重
)
model = model.to(device)
# 2. 加载你自己的 .pth 权重
ckpt_path = "/data/yjy_data/B2DiffRL/FineTuneClip/clip_finetuned_season.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model.eval()

# 3. 加载数据
json_path = "/data/yjy_data/B2DiffRL/FineTuneClip/SEN_season_JSON.json"
dataset = ImageTextJSONDataset(json_path, preprocess)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# 4. 开始评估
results = []

with torch.no_grad():
    for images, prompts, image_paths in tqdm(dataloader, desc="Evaluating CLIP"):
        images = images.to(device)
        prompts_tokenized = tokenizer(prompts).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(prompts_tokenized)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        sims = (image_features * text_features).sum(dim=-1)  # 每张图与对应文本的匹配度

        for path, prompt, score in zip(image_paths, prompts, sims.cpu().tolist()):
            results.append({
                "image": path,
                "prompt": prompt,
                "similarity": round(score, 4)
            })

# 5. 保存结果
save_path = "/data/yjy_data/B2DiffRL/FineTuneClip/clip_eval_results2.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 完成评估，相似度结果已保存至：{save_path}")
