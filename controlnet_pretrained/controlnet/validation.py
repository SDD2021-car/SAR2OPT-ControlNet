import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
import os
from diffusers import (
    UniPCMultistepScheduler,
    DDIMScheduler
)
from tqdm import tqdm
from .pipeline import StableDiffusionControlNetPipeline
# from diffusers import StableDiffusionControlNetPipeline

from diffusers.utils import check_min_version, is_wandb_available
import ipdb
from accelerate.logging import get_logger

if is_wandb_available():
    import wandb

logger = get_logger(__name__)  # pylint: disable=invalid-name

def log_validation(vae, text_encoder, tokenizer, unet, controlnet, cfg, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=cfg.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if cfg.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

    if len(cfg.validation_image) == len(cfg.validation_prompt):
        validation_images = cfg.validation_image
        validation_prompts = cfg.validation_prompt
    elif len(cfg.validation_image) == 1:
        validation_images = cfg.validation_image * len(cfg.validation_prompt)
        validation_prompts = cfg.validation_prompt
    elif len(cfg.validation_prompt) == 1:
        validation_images = cfg.validation_image
        validation_prompts = cfg.validation_prompt * len(cfg.validation_image)
    else:
        raise ValueError(
            "number of `cfg.validation_image` and `cfg.validation_prompt` should be checked in `parse_cfg`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert('RGB')

        validation_image = validation_image.resize((cfg.resolution, cfg.resolution))
        images = []

        for _ in range(cfg.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    # ipdb.set_trace()
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")



def log_validation_pipeline_unclip(pipeline, controlnet, cfg, accelerator, weight_dtype, step,):
    logger.info("Running validation... ")
    controlnet = accelerator.unwrap_model(controlnet)
    pipeline.controlnet = controlnet
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_model_cpu_offload()  # 节省显存
    pipeline.enable_attention_slicing()  # 对大图有效

    if cfg.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

    image_logs = []
    validation_prompt_info = []
    with open(cfg.validation_prompt_json_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())  # 每行一个字典
            for name, prompt in item.items():
                validation_prompt_info.append({"name": name, "prompt": prompt})
    for item in tqdm(validation_prompt_info, desc="Generating images"):
        validation_prompt = item["prompt"]
        name = item["name"]  # 不带扩展名

        # 构造图像路径
        validation_image_path = os.path.join(cfg.validation_image_dir, name + ".png")
        validation_image_embed_path = os.path.join(cfg.validation_image_embed_dir, name + ".png")
        base_name = name
        # 加载图像
        validation_image = Image.open(validation_image_path).convert("RGB")
        validation_image = validation_image.resize((cfg.resolution, cfg.resolution))

        validation_image_embed = Image.open(validation_image_embed_path).convert("RGB")


        with torch.autocast("cuda"):
            image = pipeline(
                prompt=validation_prompt,
                control_image=validation_image,
                image=validation_image_embed,
                num_inference_steps=20,
                generator=generator,
                noise_level=cfg.noise_level,
                controlnet_image_embeds_type=cfg.controlnet_image_embeds_type,
            ).images[0]

        save_path = os.path.join(cfg.output_dir, f"{base_name}.png")
        image.save(save_path)

        # image_logs.append(
        #     {"validation_image": validation_image, "validation_image_embed": validation_image_embed.resize((cfg.resolution, cfg.resolution)), "images": images, "validation_prompt": validation_prompt}
        # )



