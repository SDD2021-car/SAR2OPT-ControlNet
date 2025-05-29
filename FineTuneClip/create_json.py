# import os
# import json
#
# # 设置 source 和 target 文件夹路径
# source_folder = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/image_test"
# control_folder = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/control_image_test"
# # 设置输出的 JSON 文件路径
# output_json_path = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/SEN_season_inference1.json"
#
# # 获取 source 文件夹中的所有图像文件（假设 target 文件夹有对应同名文件）
# image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
# entries = []
#
# for filename in os.listdir(control_folder):
#     if any(filename.lower().endswith(ext) for ext in image_extensions):
#         control_path = os.path.join(control_folder, filename).replace("\\", "/")
#         source_path = os.path.join(source_folder, filename).replace("\\", "/")
#
#         # 自动根据文件名设置 prompt
#         lower_name = filename.lower()
#         if "spring" in lower_name:
#             prompt = "an image in spring"
#         elif "summer" in lower_name:
#             prompt = "an image in summer"
#         elif "fall" in lower_name or "autumn" in lower_name:
#             prompt = "an image in fall"
#         elif "winter" in lower_name:
#             prompt = "an image in winter"
#         else:
#             prompt = "an image"  # 默认 prompt
#
#         entries.append({
#             "image": source_path,
#             "control_image":control_path,
#             "prompt": prompt
#         })

# # 保存为 JSON 文件
# with open(output_json_path, "w", encoding="utf-8") as f:
#     json.dump(entries, f, ensure_ascii=False, indent=4)
#
# print(f"JSON 文件已生成：{output_json_path}")

# import json
# import os
#
# # 设置原始 JSON 文件路径和输出文件路径
# input_json_path = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/SEN_season_inference1.json"  # ← 这里换成你的原始 JSON 文件路径
# output_json_path = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/SEN_season_inference2.json"  # 修改后的输出文件
#
# # 读取原始 JSON 文件
# with open(input_json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # 修改每条数据的 source 和 target 路径，只保留文件名
# for item in data:
#     item["image"] = os.path.basename(item["image"])
#     item["control_image"] = os.path.basename(item["control_image"])
#
# # 保存修改后的 JSON 文件
# with open(output_json_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
#
# print(f"✅ 修改完成，结果已保存为：{output_json_path}")

#
# import json
#
# # 输入和输出路径
# input_path = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/SEN_season_inference2.json"     # ← 你的原始 JSON 文件路径
# output_path = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/SEN_season_inference3.json"        # ← 输出的 JSON 文件，每行一个样本
#
# # 读取 JSON 数据
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # 写入为 JSON Lines 格式（每行一个 JSON 对象）
# with open(output_path, "w", encoding="utf-8") as f:
#     for item in data:
#         json_line = json.dumps(item, ensure_ascii=False)
#         f.write(json_line + "\n")
#
# print(f"✅ 每行一个 JSON 样本，已保存为：{output_path}")

# import json
# from pathlib import Path
#
# # 输入输出文件路径
# input_file = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/SEN_season_JSON.json"
# output_file = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/SEN_season_JSON2.json"
#
# with open(input_file, "r") as fin, open(output_file, "w") as fout:
#     for line in fin:
#         data = json.loads(line.strip())
#         image_path = Path(data["image"])
#         filename = image_path.stem  # 去掉路径和扩展名，例如 "ROIs1868_summer_s1_38_p634"
#         prompt = data["prompt"]
#         fout.write(json.dumps({filename: prompt}) + "\n")

######合并版，在inference时应用#######
import os
import json
from pathlib import Path
# 设置 source 和 target 文件夹路径
source_folder = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/image_test"
control_folder = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/control_image_test"
# 设置输出的 JSON 文件路径
output_json_path = "/data/yjy_data/B2DiffRL/controlnet_pretrained/dataset/SEN_season_inference1.json"

# 获取 source 文件夹中的所有图像文件（假设 target 文件夹有对应同名文件）
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
entries = []

for filename in os.listdir(control_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        control_path = os.path.join(control_folder, filename).replace("\\", "/")
        source_path = os.path.join(source_folder, filename).replace("\\", "/")
        source_path = Path(source_path).stem
        control_path = Path(control_path).stem
        # 自动根据文件名设置 prompt
        lower_name = filename.lower()
        if "spring" in lower_name:
            prompt = "an image in spring"
        elif "summer" in lower_name:
            prompt = "an image in summer"
        elif "fall" in lower_name or "autumn" in lower_name:
            prompt = "an image in fall"
        elif "winter" in lower_name:
            prompt = "an image in winter"
        else:
            prompt = "an image"  # 默认 prompt

        entries.append({
            "image": source_path,
            "control_image":control_path,
            "prompt": prompt
        })
    with open(output_json_path, "w") as fout:
        for entry in entries:
            image_path = Path(entry["image"])
            filename = image_path.stem  # e.g., "ROIs1868_summer_s1_38_p634"
            prompt = entry["prompt"]
            fout.write(json.dumps({filename: prompt}) + "\n")
