# DreamID-Omni on Consumer GPUs — No H100 Required

Run **DreamID-Omni** (11.66B parameter identity-preserved audio-video model) on **multiple consumer 24GB GPUs** using layer sharding. This fork replaces the original sequence parallelism (which duplicates the full model on each GPU) with `accelerate`'s `dispatch_model` to **split model layers** across your cards.

### Demo: Generated on 6x NVIDIA A5000 (24GB each)

https://github.com/user-attachments/assets/luther_demo.mp4

<video src="assets/luther_demo.mp4" width="100%" controls></video>

*Martin Luther discussing the weather with Aristotle at a Victorian train station. 50 denoising steps, 512x992, ~17 minutes on consumer hardware.*

> Fork of [DreamID-Omni](https://github.com/Guoxu1233/DreamID-Omni) by Tsinghua University / ByteDance.
> Layer sharding technique from [H100 Not Required: 32B FLUX.2-dev on 2017 Hardware](https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha).

## What This Fork Adds

| Original | This Fork |
|----------|-----------|
| Needs 1x 80GB GPU (A100/H100) or 8-GPU sequence parallelism | Runs on **3-6x 24GB consumer GPUs** (RTX 3090, A5000, RTX 4090) |
| Sequence parallelism duplicates full model per GPU | Layer sharding — each GPU holds a **different slice** of the model |
| Requires flash_attn (ABI issues with newer PyTorch) | PyTorch SDPA fallback — works out of the box |
| CLI only | **Gradio UI** included |

## How It Works

```
┌────────────────────────────────────────────────────────────────┐
│  Layer Sharding: 11.66B model split across 6x 24GB GPUs       │
│                                                                │
│  GPU 0: Video+Audio Blocks 0-5 + Embeddings + VAE   ~11 GB    │
│  GPU 1: Video+Audio Blocks 6-11                      ~7 GB    │
│  GPU 2: Video+Audio Blocks 12-17                     ~7 GB    │
│  GPU 3: Video+Audio Blocks 18-23 + T5 Encoder       ~13 GB   │
│  GPU 4: Video+Audio Blocks 24-29 + Heads              ~7 GB   │
│                                                                │
│  Video block[i] and Audio block[i] stay CO-LOCATED             │
│  (they cross-attend each other — must be same GPU)             │
└────────────────────────────────────────────────────────────────┘
```

### Key Code Changes (5 files)

1. **`inference_sharded.py`** — New inference script with explicit device map (replaces `torchrun` sequence parallelism)
2. **`dreamid_omni/modules/fusion.py`** — Device-aware forward loop: moves tensors between GPUs as data flows through blocks
3. **`dreamid_omni/modules/model.py`** — RoPE frequency tensors follow data across devices
4. **`dreamid_omni/modules/attention.py`** — SDPA fallback when flash_attn is unavailable
5. **`app.py`** — Gradio web interface

## Hardware Requirements

| Config | GPUs | Total VRAM | Status |
|--------|------|------------|--------|
| 6x RTX 3090/A5000 (24GB) | 6 | 144 GB | Tested, works |
| 4x RTX 3090 (24GB) | 4 | 96 GB | Should work |
| 3x RTX 4090 (24GB) | 3 | 72 GB | Should work |
| 2x 24GB cards | 2 | 48 GB | Tight, may need reduced resolution |
| 1x A100/H100 (80GB) | 1 | 80 GB | Use original repo instead |

**All GPUs must be Ampere or newer** (compute >= 8.0). Turing cards (RTX 2080, Quadro RTX 8000) will break — no bfloat16 support.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/cronos3k/dreamid-omni-consumer-gpu.git
cd dreamid-omni-consumer-gpu

# 2. Setup
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install accelerate imageio[ffmpeg]

# 3. Download weights (~60GB)
python download_weights.py --output-dir ./ckpts
mv ckpts/DreamI_Omni ckpts/DreamID_Omni  # fix upstream typo

# 4. Run — set CUDA_VISIBLE_DEVICES to your Ampere GPUs
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python app.py
# Opens Gradio UI at http://localhost:7866

# Or headless:
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python inference_sharded.py
```

## Performance

On 6x NVIDIA A5000 (24GB each):

| Steps | Resolution | Time/clip | Notes |
|-------|-----------|-----------|-------|
| 10 | 512x992 | ~3.5 min | Fast preview |
| 50 | 512x992 | ~17 min | Full quality |

~20 seconds per denoising step. Slower than single-GPU due to inter-GPU data transfers, but it **runs** — which is the point.

## Gradio Interface

Upload a face image + voice sample, describe the scene, and generate identity-preserved video with synchronized audio. All from a web browser.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python app.py
```

## Gotchas We Hit

1. **Paired blocks must be co-located.** DreamID-Omni's FusionModel cross-attends video_block[i] with audio_block[i]. They must be on the same GPU.
2. **RoPE frequencies get stuck.** Rotary embeddings are stored as buffers that don't move with `dispatch_model`. Fix: `freqs = freqs.to(x.device)`.
3. **flash_attn ABI mismatch.** Won't compile against torch 2.10+. Solution: remove it — PyTorch's built-in SDPA works fine.
4. **Download script typo.** Official script saves to `DreamI_Omni`, engine expects `DreamID_Omni`.
5. **Turing GPUs break everything.** Exclude non-Ampere cards via `CUDA_VISIBLE_DEVICES`.

## Credits

- **[DreamID-Omni](https://github.com/Guoxu1233/DreamID-Omni)** — Original model by Xu Guo et al., Tsinghua University / ByteDance
- **Layer sharding technique** from [H100 Not Required](https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha)
- Built by [Gregor Hubert](https://huggingface.co/cronos3k)

## License

Apache 2.0 (following original DreamID-Omni license)

---
---

# Original Model Card (for reference)

*Everything below is from the [upstream DreamID-Omni repository](https://github.com/Guoxu1233/DreamID-Omni).*

---

## DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation

> [Xu Guo](https://github.com/Guoxu1233/) et al.
> Tsinghua University | Intelligent Creation Team, ByteDance

<p align="center">
<img src="assets/teaser.png" width=95%>
</p>

<p align="center">
  <a href="https://guoxu1233.github.io/DreamID-Omni/">Project Page</a> |
  <a href="https://arxiv.org/abs/2602.12160">Paper</a> |
  <a href="https://huggingface.co/XuGuo699/DreamID-Omni">Models</a>
</p>

### How to Create Prompts
Prompts use special tags to control characters and speech:
- **Subject Identity**: `<sub1>`, `<sub2>` — Represents character IPs from input images
- **Speech**: `<S>Your speech content here<E>` — Converted to speech using the character's reference audio

### Example Structure
See `test_case/oneip/captions/9.json` (single person) or `test_case/twoip/captions/20.json` (multi person) for prompt format examples.

### Acknowledgements

Built upon [Ovi](https://github.com/character-ai/Ovi), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [MMAudio](https://github.com/hkchengrex/MMAudio), [Phantom](https://github.com/Phantom-video/Phantom), [HuMo](https://github.com/Phantom-video/HuMo), [OpenHumanVid](https://github.com/fudan-generative-vision/OpenHumanVid).

### Citation

```bibtex
@misc{guo2026dreamidomni,
      title={DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation},
      author={Xu Guo and Fulong Ye and Qichao Sun and Liyang Chen and Bingchuan Li and Pengze Zhang and Jiawei Liu and Songtao Zhao and Qian He and Xiangwang Hou},
      year={2026},
      eprint={2602.12160},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.12160},
}
```
