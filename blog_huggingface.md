# Running DreamID-Omni on Consumer GPUs: Layer Sharding for the Rest of Us

**TL;DR:** We got DreamID-Omni (11.66B parameter identity-preserved audio-video model) running across 6x 24GB consumer GPUs using layer sharding. No H100 required. Here's how.

## The Problem

DreamID-Omni from Tsinghua/ByteDance is remarkable — it generates videos with preserved character identity and synchronized speech from just a face image and voice sample. But it needs ~80GB VRAM for single-GPU inference. That means A100 or H100, which most of us don't have.

What many of us *do* have: multiple 24GB cards (RTX 3090, A5000, RTX 4090). Six of them give us 144GB — more than enough if we could split the model properly.

## Why Standard Multi-GPU Doesn't Work

DreamID-Omni ships with sequence parallelism via `torchrun`. The problem? Sequence parallelism **duplicates the entire model** on each GPU. Each GPU needs to hold the full 11.66B model, which at bf16 is ~23GB. Add activations and you're at 40GB+ per GPU — right back to needing large cards.

## The Solution: Layer Sharding

Instead of duplicating, we **split the model's layers** across GPUs. The model has 30 video transformer blocks and 30 audio transformer blocks. We distribute them:

```
GPU 0: Blocks 0-5 + Embeddings + VAE    (~5GB weights, ~11GB with activations)
GPU 1: Blocks 6-11                       (~5GB weights)
GPU 2: Blocks 12-17                      (~5GB weights)
GPU 3: Blocks 18-23 + T5 Text Encoder    (~5GB + 8GB)
GPU 4: Blocks 24-29 + Output Heads       (~5GB weights)
```

Total model memory: ~23GB spread across 5 GPUs. Each card uses 7-13GB, leaving plenty of headroom for activations.

## The Tricky Part: Paired Cross-Attention

DreamID-Omni isn't a simple sequential model. Its `FusionModel` interleaves video and audio blocks — `video_block[i]` and `audio_block[i]` cross-attend each other at every layer. This means paired blocks **must be on the same GPU**.

Our device map ensures this:
```python
for i in range(30):
    gpu_idx = gpu_list[i // blocks_per_gpu]
    device_map[f"video_model.blocks.{i}"] = gpu_idx
    device_map[f"audio_model.blocks.{i}"] = gpu_idx  # Same GPU!
```

Then in the forward loop, we move data between GPUs as it flows through the layers:
```python
for i in range(self.num_blocks):
    block_device = next(vid_block.parameters()).device
    if vid.device != block_device:
        vid = vid.to(block_device)
        audio = audio.to(block_device)
        kwargs = {k: v.to(block_device) if isinstance(v, torch.Tensor) else v
                  for k, v in kwargs.items()}
```

## Other Gotchas We Hit

**1. RoPE frequencies stuck on wrong GPU.** Rotary position embeddings are stored as buffers during `set_rope_params()`. When `dispatch_model` places layers across GPUs, these buffers don't automatically follow. Fix: `freqs = freqs.to(x.device)` in the RoPE function.

**2. Flash Attention ABI mismatch.** The `flash_attn` package wouldn't compile against torch 2.10. Solution: just remove it — PyTorch's built-in `scaled_dot_product_attention` works as a drop-in fallback (the codebase already had the fallback path, just not in `flash_attention()` directly).

**3. Turing GPUs break everything.** Our RTX 8000 (compute 7.5) caused global failures — bfloat16 and Flash Attention need compute >= 8.0. Always exclude non-Ampere cards via `CUDA_VISIBLE_DEVICES`.

**4. Download script typo.** The official `download_weights.py` saves to `DreamI_Omni` but the engine looks for `DreamID_Omni`. A `mv` fixes it.

## Performance

On 6x NVIDIA A5000 (24GB each):

| Steps | Resolution | Time/clip | Quality |
|-------|-----------|-----------|---------|
| 10 | 512x992 | ~3.5 min | Preview |
| 50 | 512x992 | ~17 min | Full |

~20 seconds per denoising step. The inter-GPU data transfer adds overhead compared to single-GPU, but it *runs* — which is the point.

## Try It

```bash
git clone https://github.com/cronos3k/dreamid-omni-consumer-gpu.git
cd dreamid-omni-consumer-gpu
# Setup, download weights, then:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python app.py
```

Full instructions in the [README](https://github.com/cronos3k/dreamid-omni-consumer-gpu).

This technique generalizes to any model with a block-sequential architecture. If you can identify paired/grouped layers, you can shard them across consumer cards. We previously used the same approach for [32B FLUX.2-dev on 2017 Tesla V100s](https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha).

The AI hardware moat is thinner than you think.

---

*Built by Gregor Hubert ([@cronos3k](https://huggingface.co/cronos3k)). DreamID-Omni by Tsinghua University / ByteDance.*
