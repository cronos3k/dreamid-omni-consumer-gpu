#!/usr/bin/env python3
"""DreamID-Omni Sharded Inference — split model layers across multiple 24GB GPUs.

Based on the technique from:
  https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha

Instead of sequence parallelism (which duplicates the model on each GPU),
this shards the model LAYERS across GPUs so no single GPU needs the full model.

GPU Layout (Ampere only, exclude GPU 5):
  GPU 0 (3090 24GB):  FusionModel layers 0-9
  GPU 1 (A5000 24GB): FusionModel layers 10-19
  GPU 3 (3090 24GB):  FusionModel layers 20-29
  GPU 4 (A5000 24GB): T5 text encoder
  GPU 6 (A5000 24GB): FusionModel overflow + VAEs
  GPU 7 (A5000 24GB): FusionModel overflow

  Total: ~23GB model in bf16 + VAEs + T5 across 6x 24GB = 144GB available
"""

import os
import sys
import logging
import json
import re
import time
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

# Must be set before any CUDA init
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,6,7"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dreamid_omni.utils.io_utils import save_video
from dreamid_omni.utils.model_loading_utils import (
    init_fusion_score_model_ovi, init_text_model,
    init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint
)
from dreamid_omni.modules.fusion import FusionModel


def load_test_data(data_root="./test_case"):
    """Load test cases from the data directory."""
    all_data = []
    modes = {
        'oneip': {'img_folder': 'imgs', 'audio_folder': 'audios', 'json_folder': 'captions'},
        'twoip': {'img_folder': 'imgs', 'audio_folder': 'audios', 'json_folder': 'captions'}
    }
    for mode, subfolders in modes.items():
        mode_path = os.path.join(data_root, mode)
        if not os.path.exists(mode_path):
            continue
        img_root = os.path.join(mode_path, subfolders['img_folder'])
        audio_root = os.path.join(mode_path, subfolders['audio_folder'])
        json_root = os.path.join(mode_path, subfolders['json_folder'])
        if not os.path.exists(img_root):
            continue
        for test_name in os.listdir(img_root):
            text_prompt_path = os.path.join(json_root, f'{test_name}.json')
            if mode == 'oneip':
                image0_path = os.path.join(img_root, test_name, f'{test_name}.png')
                image1_path = None
                audio0_path = os.path.join(audio_root, test_name, f'{test_name}.wav')
                audio1_path = None
                check_paths = [text_prompt_path, image0_path, audio0_path]
            else:
                image0_path = os.path.join(img_root, test_name, '0.png')
                image1_path = os.path.join(img_root, test_name, '1.png')
                audio0_path = os.path.join(audio_root, test_name, '0.wav')
                audio1_path = os.path.join(audio_root, test_name, '1.wav')
                check_paths = [text_prompt_path, image0_path, image1_path, audio0_path, audio1_path]
            if not all(os.path.exists(p) for p in check_paths):
                continue
            try:
                with open(text_prompt_path, 'r', encoding='utf-8') as f:
                    text_prompt = json.load(f)
                text_prompt = re.sub(r'\[SPEAKER_TIMESTAMPS_START\].*?\[SPEAKER_TIMESTAMPS_END\]', '', text_prompt, flags=re.DOTALL).strip()
                text_prompt = re.sub(r'\[AUDIO_DESCRIPTION_START].*?\[AUDIO_DESCRIPTION_END]', '', text_prompt, flags=re.DOTALL).strip()
                text_prompt = re.sub(r'\[[A-Z_]+\]', '', text_prompt)
                text_prompt = re.sub(r'\n\s*\n', '\n', text_prompt).strip()
            except Exception:
                continue
            all_data.append((f"{mode}_{test_name}", text_prompt, image0_path, image1_path, audio0_path, audio1_path))
    return all_data


def main():
    config = OmegaConf.load('dreamid_omni/configs/inference/inference_r2av_local.yaml')
    target_dtype = torch.bfloat16

    logging.info("=" * 60)
    logging.info("DreamID-Omni — SHARDED Multi-GPU Inference")
    logging.info(f"GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logging.info("=" * 60)

    # ── Step 1: Load FusionModel on meta device, then shard ──────────
    logging.info("Initializing FusionModel on meta device...")
    model, video_config, audio_config = init_fusion_score_model_ovi(rank=0, meta_init=True)

    # Compute device map WHILE model is still on meta (before materializing)
    # Reserve GPUs 0-3 for the fusion model (4x 24GB = 96GB)
    # GPU 4 for T5, GPU 5 for VAEs
    logging.info("Computing device map on meta model...")
    # Build explicit device map — video block i and audio block i MUST be
    # on the same GPU because FusionModel interleaves them with cross-attention.
    # 30 block pairs across 5 GPUs = 6 pairs per GPU
    # Each pair ≈ 380MB (video) + 380MB (audio) = 760MB bf16
    # 6 pairs = ~4.5 GB per GPU — plenty of room on 24GB cards
    device_map = {}

    # Embeddings on GPU 0
    for prefix in ["video_model", "audio_model"]:
        device_map[f"{prefix}.patch_embedding"] = 0
        device_map[f"{prefix}.text_embedding"] = 0
        device_map[f"{prefix}.time_embedding"] = 0
        device_map[f"{prefix}.time_projection"] = 0

    # 30 block pairs split across GPUs 0,1,2,4,5 (6 pairs each)
    gpu_list = [0, 1, 2, 4, 5]
    blocks_per_gpu = 6
    for i in range(30):
        gpu_idx = gpu_list[i // blocks_per_gpu]
        device_map[f"video_model.blocks.{i}"] = gpu_idx
        device_map[f"audio_model.blocks.{i}"] = gpu_idx

    # Heads on last GPU
    device_map["video_model.head"] = 5
    device_map["audio_model.head"] = 5
    logging.info(f"Device map ({len(device_map)} entries):")
    # Log a summary of which devices get what
    dev_counts = {}
    for k, v in device_map.items():
        dev_counts[v] = dev_counts.get(v, 0) + 1
    for dev, count in sorted(dev_counts.items(), key=lambda x: str(x[0])):
        logging.info(f"  Device {dev}: {count} modules")

    # Load checkpoint and dispatch to devices
    checkpoint_path = os.path.join(config.ckpt_dir, "DreamID_Omni", "dreamid_omni.safetensors")
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=True)
    model = model.to(dtype=target_dtype).eval()
    model.set_rope_params()

    logging.info("Dispatching model across GPUs...")
    model = dispatch_model(model, device_map=device_map)
    logging.info("FusionModel sharded across GPUs!")

    # ── Step 2: Load VAEs on GPU 0 ──────────────────────────────────
    logging.info("Loading Video VAE...")
    vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=0)
    vae_model_video.model.requires_grad_(False).eval()
    vae_model_video.model = vae_model_video.model.bfloat16()

    logging.info("Loading Audio VAE...")
    vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=0)
    vae_model_audio.requires_grad_(False).eval()
    vae_model_audio = vae_model_audio.bfloat16()

    # ── Step 3: Load T5 on GPU 3 (visible idx 3 = physical GPU 4) ───
    logging.info("Loading T5 text encoder...")
    text_model = init_text_model(config.ckpt_dir, rank=3, cpu_offload=False)

    logging.info("All models loaded!")

    # ── Step 4: Run inference ────────────────────────────────────────
    all_data = load_test_data(config.get("data_root", "./test_case"))
    logging.info(f"Found {len(all_data)} test cases")

    output_dir = config.get("output_dir", "./result/sharded_test")
    os.makedirs(output_dir, exist_ok=True)

    # We need to manually run the generate pipeline since the standard
    # engine assumes single-device placement. This is the core loop
    # adapted from dreamid_omni_engine.py generate()

    from dreamid_omni.dreamid_omni_engine import DreamIDOmniEngine

    # Create a lightweight engine wrapper with our sharded model
    engine = DreamIDOmniEngine.__new__(DreamIDOmniEngine)
    engine.model = model
    engine.vae_model_video = vae_model_video
    engine.vae_model_audio = vae_model_audio
    engine.text_model = text_model
    engine.device = 0  # VAE device
    engine.target_dtype = target_dtype
    engine.cpu_offload = False  # Model is already sharded, no offloading needed
    engine.model_name = "dreamid_omni"
    engine.video_latent_length = 31
    engine.audio_latent_length = 157
    engine.image_model = None
    engine.audio_latent_channel = audio_config.get("in_dim", 128)
    engine.video_latent_channel = video_config.get("in_dim", 16)

    for test_name, text_prompt, image0_path, image1_path, audio0_path, audio1_path in tqdm(all_data):
        logging.info(f"Processing: {test_name}")

        hw = config.get("video_frame_height_width", [512, 992])
        seed = config.get("seed", 100)

        output_path = os.path.join(output_dir, f"{test_name}_{'x'.join(map(str, hw))}_{seed}.mp4")
        if os.path.exists(output_path):
            logging.info(f"  Skipping (already exists)")
            continue

        try:
            result = engine.generate(
                text_prompt=text_prompt,
                image0_path=image0_path,
                image1_path=image1_path,
                audio0_path=audio0_path,
                audio1_path=audio1_path,
                video_frame_height_width=hw,
                seed=seed,
                solver_name=config.get("solver_name", "unipc"),
                sample_steps=config.get("sample_steps", 50),
                shift=config.get("shift", 5.0),
                video_cfg_scale=config.get("video_cfg_scale", 3.0),
                video_ref_cfg_scale=config.get("video_ref_cfg_scale", 1.5),
                audio_cfg_scale=config.get("audio_cfg_scale", 4.0),
                audio_ref_cfg_scale=config.get("audio_ref_cfg_scale", 2.0),
                video_negative_prompt=config.get("video_negative_prompt", ""),
                audio_negative_prompt=config.get("audio_negative_prompt", ""),
            )

            if result is not None:
                generated_video, generated_audio, _ = result
                # Convert to numpy if tensor, squeeze batch dim if present
                import numpy as np
                if hasattr(generated_video, 'cpu'):
                    generated_video = generated_video.cpu().float().numpy()
                if generated_video.ndim == 5:
                    generated_video = generated_video[0]  # remove batch dim
                if hasattr(generated_audio, 'cpu'):
                    generated_audio = generated_audio.cpu().float().numpy()
                if generated_audio is not None and generated_audio.ndim > 1:
                    generated_audio = generated_audio.squeeze()
                logging.info(f"  Video shape: {generated_video.shape}, dtype: {generated_video.dtype}")
                logging.info(f"  Audio shape: {generated_audio.shape if generated_audio is not None else None}")
                # Save directly — bypass save_video's strict checks
                import imageio
                from scipy.io import wavfile
                # (C, F, H, W) -> (F, H, W, C)
                vid_np = generated_video.transpose(1, 2, 3, 0)
                if vid_np.max() <= 1.0:
                    vid_np = np.clip(vid_np, -1, 1)
                    vid_np = ((vid_np + 1) / 2 * 255).astype(np.uint8)
                else:
                    vid_np = vid_np.astype(np.uint8)
                writer = imageio.get_writer(output_path, fps=24, codec='libx264')
                for frame in vid_np:
                    writer.append_data(frame)
                writer.close()
                # Save audio
                if generated_audio is not None:
                    audio_path = output_path.replace('.mp4', '.wav')
                    audio_int16 = (np.clip(generated_audio, -1, 1) * 32767).astype(np.int16)
                    wavfile.write(audio_path, 16000, audio_int16)
                    logging.info(f"  Saved: {output_path} + {audio_path}")
                else:
                    logging.info(f"  Saved: {output_path}")
            else:
                logging.error(f"  Generation returned None")

        except Exception as e:
            logging.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    logging.info("Done!")


if __name__ == "__main__":
    main()
