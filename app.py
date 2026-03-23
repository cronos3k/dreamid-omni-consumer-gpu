#!/usr/bin/env python3
"""DreamID-Omni Sharded — Gradio Interface

Run identity-preserved audio-video generation across multiple consumer GPUs
using layer sharding. No H100/A100 required.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 python app.py
"""

import os
import sys
import logging
import json
import re
import time
import tempfile
import uuid
from pathlib import Path
from datetime import datetime

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import numpy as np
import gradio as gr
from accelerate import dispatch_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dreamid_omni.utils.model_loading_utils import (
    init_fusion_score_model_ovi, init_text_model,
    init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint
)
from dreamid_omni.dreamid_omni_engine import DreamIDOmniEngine

# ── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./result/gradio_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_VISIBLE_GPUS = torch.cuda.device_count()
TARGET_DTYPE = torch.bfloat16

# ── Model Loading ────────────────────────────────────────────────────────────

engine = None


def get_gpu_info():
    """Get VRAM usage for all visible GPUs."""
    lines = []
    for i in range(NUM_VISIBLE_GPUS):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / 1e9
        total_gb = total / 1e9
        lines.append(f"GPU {i}: {name} — {used:.1f}/{total_gb:.1f} GB")
    return "\n".join(lines)


def build_device_map(model, num_gpus, t5_gpu=None):
    """Build explicit device map pairing video+audio blocks on same GPU.

    The FusionModel has video_model (30 blocks) and audio_model (30 blocks).
    Paired blocks must be co-located because they cross-attend each other.
    """
    device_map = {}

    # Reserve one GPU for T5 if specified
    if t5_gpu is not None:
        model_gpus = [i for i in range(num_gpus) if i != t5_gpu]
    else:
        model_gpus = list(range(num_gpus))

    # Embeddings on first GPU
    for prefix in ["video_model", "audio_model"]:
        device_map[f"{prefix}.patch_embedding"] = model_gpus[0]
        device_map[f"{prefix}.text_embedding"] = model_gpus[0]
        device_map[f"{prefix}.time_embedding"] = model_gpus[0]
        device_map[f"{prefix}.time_projection"] = model_gpus[0]

    # 30 block pairs across available GPUs
    blocks_per_gpu = max(1, 30 // len(model_gpus))
    for i in range(30):
        gpu_idx = model_gpus[min(i // blocks_per_gpu, len(model_gpus) - 1)]
        device_map[f"video_model.blocks.{i}"] = gpu_idx
        device_map[f"audio_model.blocks.{i}"] = gpu_idx

    # Heads on last GPU
    device_map["video_model.head"] = model_gpus[-1]
    device_map["audio_model.head"] = model_gpus[-1]

    return device_map


def load_engine():
    """Load DreamID-Omni with layer sharding across available GPUs."""
    global engine

    if engine is not None:
        return "Already loaded"

    ckpt_dir = "./ckpts"
    t5_gpu = min(NUM_VISIBLE_GPUS - 1, 3)  # T5 on a middle GPU

    logging.info(f"Loading DreamID-Omni across {NUM_VISIBLE_GPUS} GPUs...")

    # 1. Init model on meta device
    model, video_config, audio_config = init_fusion_score_model_ovi(rank=0, meta_init=True)

    # 2. Compute device map
    device_map = build_device_map(model, NUM_VISIBLE_GPUS, t5_gpu=t5_gpu)
    dev_counts = {}
    for v in device_map.values():
        dev_counts[v] = dev_counts.get(v, 0) + 1
    for dev, count in sorted(dev_counts.items()):
        logging.info(f"  GPU {dev}: {count} modules")

    # 3. Load weights and dispatch
    checkpoint_path = os.path.join(ckpt_dir, "DreamID_Omni", "dreamid_omni.safetensors")
    load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=True)
    model = model.to(dtype=TARGET_DTYPE).eval()
    model.set_rope_params()
    model = dispatch_model(model, device_map=device_map)

    # 4. Load VAEs on GPU 0
    vae_model_video = init_wan_vae_2_2(ckpt_dir, rank=0)
    vae_model_video.model.requires_grad_(False).eval()
    vae_model_video.model = vae_model_video.model.bfloat16()

    vae_model_audio = init_mmaudio_vae(ckpt_dir, rank=0)
    vae_model_audio.requires_grad_(False).eval()
    vae_model_audio = vae_model_audio.bfloat16()

    # 5. Load T5
    text_model = init_text_model(ckpt_dir, rank=t5_gpu, cpu_offload=False)

    # 6. Assemble engine
    engine = DreamIDOmniEngine.__new__(DreamIDOmniEngine)
    engine.model = model
    engine.vae_model_video = vae_model_video
    engine.vae_model_audio = vae_model_audio
    engine.text_model = text_model
    engine.device = 0
    engine.target_dtype = TARGET_DTYPE
    engine.cpu_offload = False
    engine.model_name = "dreamid_omni"
    engine.video_latent_length = 31
    engine.audio_latent_length = 157
    engine.image_model = None
    engine.audio_latent_channel = audio_config.get("in_dim", 128)
    engine.video_latent_channel = video_config.get("in_dim", 16)

    logging.info("DreamID-Omni loaded and sharded!")
    return f"Loaded across {NUM_VISIBLE_GPUS} GPUs"


# ── Generation ───────────────────────────────────────────────────────────────


def generate_video(
    face_image, voice_audio, prompt,
    steps, video_cfg, video_ref_cfg, audio_cfg, audio_ref_cfg,
    seed, resolution,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate identity-preserved video with synced audio."""
    if engine is None:
        raise gr.Error("Model not loaded. Click 'Load Model' first.")
    if face_image is None:
        raise gr.Error("Upload a face image.")
    if voice_audio is None:
        raise gr.Error("Upload a voice audio sample.")
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")

    # Save inputs to temp files
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = OUTPUT_DIR / f"{ts}_{str(uuid.uuid4())[:6]}"
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save face image
    from PIL import Image
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(face_image)
    face_path = str(job_dir / "face.png")
    face_image.save(face_path)

    # Save audio
    import scipy.io.wavfile as wavfile
    audio_path = str(job_dir / "voice.wav")
    if isinstance(voice_audio, tuple):
        sr, audio_data = voice_audio
        wavfile.write(audio_path, sr, audio_data)
    elif isinstance(voice_audio, str):
        import shutil
        shutil.copy(voice_audio, audio_path)

    # Parse resolution
    hw = [int(x) for x in resolution.split("x")]

    # Build structured prompt
    structured_prompt = (
        f"[IMAGE_CAPTIONS_START]\n"
        f"<img1>: {prompt}\n"
        f"[IMAGE_CAPTIONS_END]\n\n"
        f"[FINAL_VIDEO_CAPTION_START]\n"
        f"[VISUAL_DESCRIPTION_START]\n"
        f"{prompt}\n"
        f"[VISUAL_DESCRIPTION_END]\n"
        f"[CO_ANALYSIS_START]\n"
        f"<sub1> speaks naturally.\n"
        f"[CO_ANALYSIS_END]\n"
        f"[FINAL_VIDEO_CAPTION_END]"
    )

    t0 = time.time()

    try:
        result = engine.generate(
            text_prompt=structured_prompt,
            image0_path=face_path,
            image1_path=None,
            audio0_path=audio_path,
            audio1_path=None,
            video_frame_height_width=hw,
            seed=int(seed),
            solver_name="unipc",
            sample_steps=int(steps),
            shift=5.0,
            video_cfg_scale=video_cfg,
            video_ref_cfg_scale=video_ref_cfg,
            audio_cfg_scale=audio_cfg,
            audio_ref_cfg_scale=audio_ref_cfg,
            video_negative_prompt="jitter, bad hands, blur, distortion",
            audio_negative_prompt="robotic, muffled, echo, distorted",
        )
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")

    elapsed = time.time() - t0

    if result is None:
        raise gr.Error("Generation returned None — check logs")

    generated_video, generated_audio, _ = result

    # Convert to numpy
    if hasattr(generated_video, 'cpu'):
        generated_video = generated_video.cpu().float().numpy()
    if generated_video.ndim == 5:
        generated_video = generated_video[0]
    if hasattr(generated_audio, 'cpu'):
        generated_audio = generated_audio.cpu().float().numpy()
    if generated_audio is not None and generated_audio.ndim > 1:
        generated_audio = generated_audio.squeeze()

    # Save video
    import imageio
    output_path = str(job_dir / "output.mp4")
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
        wav_path = str(job_dir / "output.wav")
        audio_int16 = (np.clip(generated_audio, -1, 1) * 32767).astype(np.int16)
        wavfile.write(wav_path, 16000, audio_int16)

    # Combine video + audio with ffmpeg
    final_path = str(job_dir / "final.mp4")
    if generated_audio is not None:
        os.system(f'ffmpeg -y -i {output_path} -i {wav_path} -c:v copy -c:a aac -shortest {final_path} 2>/dev/null')
        if not os.path.exists(final_path):
            final_path = output_path
    else:
        final_path = output_path

    info = (
        f"{elapsed:.0f}s | {int(steps)} steps | {resolution} | "
        f"seed {int(seed)} | {NUM_VISIBLE_GPUS} GPUs | "
        f"saved: {job_dir.name}"
    )
    return final_path, info


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="DreamID-Omni Sharded") as demo:

        gr.Markdown("# DreamID-Omni — Multi-GPU Sharded Inference")
        gr.Markdown(
            f"**{NUM_VISIBLE_GPUS} GPUs detected** | "
            "Identity-preserved video with synced audio | "
            "No H100 required — runs on consumer GPUs via layer sharding"
        )

        with gr.Row():
            load_btn = gr.Button("Load Model", variant="primary")
            load_status = gr.Textbox(value="Not loaded", label="Status", interactive=False)
        gpu_info = gr.Textbox(value=get_gpu_info(), label="GPU Status", lines=NUM_VISIBLE_GPUS, interactive=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                face_img = gr.Image(type="pil", label="Face Reference Image", height=250)
                voice_audio = gr.Audio(label="Voice Sample (WAV)", type="filepath")
                prompt = gr.Textbox(
                    label="Scene Description",
                    placeholder="A young woman sitting in a cafe, speaking naturally...",
                    lines=4,
                )
                with gr.Row():
                    steps = gr.Slider(5, 50, value=10, step=1, label="Steps (10=fast, 50=quality)")
                    seed = gr.Number(value=42, label="Seed", precision=0)
                resolution = gr.Dropdown(
                    choices=["512x992", "704x1280"],
                    value="512x992", label="Resolution",
                )
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        video_cfg = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Video CFG")
                        video_ref_cfg = gr.Slider(0.5, 5.0, value=1.5, step=0.5, label="Video Ref CFG")
                    with gr.Row():
                        audio_cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="Audio CFG")
                        audio_ref_cfg = gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="Audio Ref CFG")
                gen_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Generated Video", height=450)
                info_text = gr.Textbox(label="Info", interactive=False, lines=2)

        # Events
        load_btn.click(load_engine, outputs=[load_status]).then(
            get_gpu_info, outputs=[gpu_info]
        )
        gen_btn.click(
            generate_video,
            inputs=[face_img, voice_audio, prompt,
                    steps, video_cfg, video_ref_cfg, audio_cfg, audio_ref_cfg,
                    seed, resolution],
            outputs=[output_video, info_text],
        ).then(get_gpu_info, outputs=[gpu_info])

        gr.Markdown("""
### How It Works
This runs **DreamID-Omni** (11.66B params) across multiple consumer GPUs using **layer sharding**:
- Video+Audio transformer blocks are split across GPUs (paired blocks stay co-located for cross-attention)
- T5 text encoder on a dedicated GPU
- VAEs on the first GPU
- No model duplication — each GPU holds a different slice of the model

Based on the technique from [H100 Not Required](https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha).
""")

        return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7866)
