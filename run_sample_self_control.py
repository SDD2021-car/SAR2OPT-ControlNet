import argparse
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

from controlnet_pretrained.controlnet.validation import log_validation_pipeline_unclip
from controlnet_pretrained.controlnet.pipeline import StableDiffusionControlNetUnCLIPPipeline
from controlnet_pretrained.controlnet.args_parser import DictAction, config_merge_dict
from controlnet_pretrained.controlnet.dataset import collate_fn_embed, ControlNetUnCLIPDataset
from controlnet_pretrained.controlnet.utils import import_model_class_from_model_name_or_path, get_full_repo_name
import copy

check_min_version("0.15.0.dev0")

logger = get_logger(__name__)


def main(cfg):
    torch.cuda.set_device(cfg.dev_id)
    logging_dir = Path(cfg.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=cfg.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Load the tokenizer
    if cfg.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, revision=cfg.revision, use_fast=False)
    elif cfg.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=cfg.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(cfg.pretrained_model_name_or_path, cfg.revision)

    # image encoding components
    feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path+"/feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path+"/image_encoder")
    # image noising components
    image_normalizer = StableUnCLIPImageNormalizer.from_pretrained(cfg.pretrained_model_name_or_path+"/image_normalizer")
    image_noising_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path+"/image_noising_scheduler")

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        cfg.pretrained_model_name_or_path+"/text_encoder", revision=cfg.revision
    )
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path+"/vae", revision=cfg.revision)
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path+"/unet", revision=cfg.revision
    )

    if cfg.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet_path, torch_dtype=torch.float16)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    validation_pipeline = StableDiffusionControlNetUnCLIPPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        image_normalizer=image_normalizer,
        image_noising_scheduler=image_noising_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if cfg.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    test_dataset = ControlNetUnCLIPDataset(cfg, tokenizer, feature_extractor)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        collate_fn=collate_fn_embed,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )


    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader = accelerator.prepare(
        controlnet, optimizer, test_dataloader
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # ipdb.set_trace()
    # if accelerator.is_main_process:
    #     # ipdb.set_trace()
    #     # tracker_config = dict(vars(cfg)) ## for args use this
    #     tracker_config = dict(copy.deepcopy(cfg))  ## for dict cfg use this
    #     # tensorboard cannot handle list types for config
    #     tracker_config.pop("validation_prompt")
    #     tracker_config.pop("validation_image")
    #     tracker_config.pop("validation_image_embed")
    #
    #     accelerator.init_trackers(cfg.tracker_project_name, config=tracker_config)

    global_step = 0

    with accelerator.accumulate(controlnet):

        if accelerator.sync_gradients:
            params_to_clip = controlnet.parameters()
            accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=cfg.set_grads_to_none)

        log_validation_pipeline_unclip(
            # logger,
            validation_pipeline,
            controlnet,
            cfg,
            accelerator,
            weight_dtype,
            global_step,
        )

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(cfg.output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./controlnet_pretrained/ControlNet_Pretrained_SEN12_season.yaml")
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,  ##NOTE cannot support multi-level config change
        help="--options is deprecated in favor of --cfg_options' and it will "
             'not be supported in version v0.22.0. Override some settings in the '
             'used config, the key-value pair in xxx=yyy format will be merged '
             'into config file. If the value to be overwritten is a list, it '
             'should be like key="[a,b]" or key=a,b It also allows nested '
             'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
             'marks are necessary and that no white space is allowed.')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)