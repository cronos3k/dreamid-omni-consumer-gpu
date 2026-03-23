# DreamID-Omni: Sharded Multi-GPU Inference

**Run DreamID-Omni (11.66B params) on consumer GPUs — no H100 required.**

This fork adds layer-sharded inference that splits the model across multiple 24GB GPUs using `accelerate`'s `dispatch_model`. The key insight: video and audio transformer blocks are paired on the same GPU (they cross-attend), while block groups are distributed across cards.

## What Changed

Three files modified from the original DreamID-Omni:

1. **`inference_sharded.py`** — Layer-sharded inference script replacing `torchrun` sequence parallelism
2. **`dreamid_omni/modules/fusion.py`** — Device-aware forward loop (moves tensors between GPUs during block iteration)
3. **`dreamid_omni/modules/model.py`** — RoPE frequency tensors follow data across devices
4. **`dreamid_omni/modules/attention.py`** — SDPA fallback when flash_attn unavailable
5. **`app.py`** — Gradio interface for interactive use

## Hardware Requirements

| Config | GPUs | Total VRAM | Status |
|--------|------|------------|--------|
| 6x RTX 3090/A5000 (24GB) | 6 | 144 GB | Tested |
| 4x RTX 3090 (24GB) | 4 | 96 GB | Should work |
| 3x RTX 4090 (24GB) | 3 | 72 GB | Should work |
| 2x RTX 4090 (24GB) | 2 | 48 GB | Tight, may need reduced resolution |
| 1x A6000 (48GB) | 1 | 48 GB | OOM — model + activations exceed 48GB |
| 1x A100/H100 (80GB) | 1 | 80 GB | Use original `inference_r2av.py` |

**Minimum: 3x 24GB Ampere GPUs** (72 GB total). More GPUs = more headroom for activations.

**Important:** All GPUs must be Ampere or newer (compute capability >= 8.0). Turing cards (RTX 2080, Quadro RTX 8000, etc.) will break — they don't support bfloat16 or Flash Attention 2.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/cronos3k/dreamid-omni-consumer-gpu.git
cd dreamid-omni-consumer-gpu
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install accelerate imageio[ffmpeg]

# 2. Download weights
python download_weights.py --output-dir ./ckpts
# Fix directory name (upstream typo)
mv ckpts/DreamI_Omni ckpts/DreamID_Omni 2>/dev/null

# 3. Run with Gradio UI (adjust CUDA_VISIBLE_DEVICES for your GPUs)
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python app.py

# Or headless inference:
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python inference_sharded.py
```

## How Layer Sharding Works

Instead of sequence parallelism (which **duplicates** the full model on each GPU), we **split model layers** across GPUs:

```
┌─────────────────────────────────────────────────────────────┐
│ Original: Sequence Parallelism (needs full model per GPU)   │
│                                                             │
│  GPU 0: [Full 11.66B model] ──┐                            │
│  GPU 1: [Full 11.66B model] ──┤── split sequence dim       │
│  GPU 2: [Full 11.66B model] ──┘                            │
│  = 3x model copies = ~70GB minimum                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Ours: Layer Sharding (splits model, no duplication)         │
│                                                             │
│  GPU 0: [Blocks 0-5 + Embeddings + VAE]  ~5GB weights      │
│  GPU 1: [Blocks 6-11]                    ~5GB weights       │
│  GPU 2: [Blocks 12-17]                   ~5GB weights       │
│  GPU 3: [Blocks 18-23 + T5]             ~5GB + 8GB         │
│  GPU 4: [Blocks 24-29 + Heads]           ~5GB weights       │
│  = 1x model copy spread across GPUs = ~23GB total           │
└─────────────────────────────────────────────────────────────┘
```

The critical detail: DreamID-Omni's FusionModel interleaves video and audio blocks with cross-attention. **Paired blocks (video_block[i] + audio_block[i]) must be on the same GPU.** Our device map ensures this.

### Key Modifications

**1. Device-aware fusion forward loop** (`fusion.py`):
```python
for i in range(self.num_blocks):
    vid_block = self.video_model.blocks[i]
    audio_block = self.audio_model.blocks[i]

    # Move data to current block's GPU
    block_device = next(vid_block.parameters()).device
    if vid.device != block_device:
        vid = vid.to(block_device)
        audio = audio.to(block_device)
        kwargs = {k: v.to(block_device) if isinstance(v, torch.Tensor) else v
                  for k, v in kwargs.items()}
```

**2. RoPE frequency device tracking** (`model.py`):
```python
freqs = freqs.to(x.device)  # Follow data across GPUs
```

**3. SDPA fallback** (`attention.py`):
```python
if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
    # PyTorch native scaled_dot_product_attention
    x = torch.nn.functional.scaled_dot_product_attention(...)
```

## Performance

| Config | Steps | Resolution | Time/clip | Quality |
|--------|-------|-----------|-----------|---------|
| 6x A5000 24GB | 50 | 512x992 | ~17 min | Full |
| 6x A5000 24GB | 10 | 512x992 | ~3.5 min | Preview |
| 6x A5000 24GB | 50 | 704x1280 | ~25 min | Full HD |

~20 seconds per denoising step. Speed is limited by inter-GPU data transfers during the forward loop.

## Limitations

- Slower than single-GPU (data moves between GPUs each block iteration)
- All GPUs must be Ampere+ (compute >= 8.0)
- Flash Attention 2 may have compatibility issues with torch 2.10+ — SDPA fallback works fine
- Video + audio are generated together (no video-only mode)

## Credits

- [DreamID-Omni](https://github.com/Guoxu1233/DreamID-Omni) — original model by Tsinghua/ByteDance
- Layer sharding technique from [H100 Not Required](https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha)
- Built by Gregor Hubert ([@cronos3k](https://huggingface.co/cronos3k))

## License

Apache 2.0 (following original DreamID-Omni license)
