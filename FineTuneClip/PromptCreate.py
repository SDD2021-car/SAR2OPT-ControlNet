import json

# 原始输入 JSON 文件
input_json_path = "/data/yjy_data/B2DiffRL/FineTuneClip/SEN_season_JSON.json"  # 替换为你的路径
# 输出的 prompt.json 文件
output_json_path = "/data/yjy_data/B2DiffRL/FineTuneClip/Prompt.json"

# 加载整个 JSON 数组
with open(input_json_path, 'r') as f:
    data = json.load(f)

# 提取 prompt 字段
prompts = [entry["prompt"] for entry in data]

# 保存到新 JSON 文件中
with open(output_json_path, 'w') as f:
    json.dump(prompts, f, indent=4)

print(f"Extracted {len(prompts)} prompts to {output_json_path}")
