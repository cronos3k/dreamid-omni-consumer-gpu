# LinkedIn Post

---

**You don't need an H100 to run 12B parameter video AI models.**

We just got DreamID-Omni — an 11.66B parameter identity-preserved video generation model from Tsinghua/ByteDance — running on six RTX A5000s (24GB each). No A100. No H100. No cloud rental.

The trick: layer sharding. Instead of duplicating the model on each GPU (the default multi-GPU approach), we split its 60 transformer blocks across GPUs. Each card holds ~5GB of weights, leaving 19GB for activations. The model's video and audio blocks cross-attend each other, so paired blocks stay co-located on the same GPU.

Three code changes made this work:
1. Explicit device map pairing video+audio block[i] on the same GPU
2. Moving shared tensors (RoPE frequencies, conditioning embeddings) to follow data across devices
3. PyTorch SDPA fallback replacing flash_attn (ABI compatibility with torch 2.10)

Result: ~20s/step, ~17 min for a 5-second clip at 512x992 with 50 denoising steps.

This is the same approach I used for running 32B FLUX.2-dev on eight 2017 Tesla V100s (https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha). The pattern generalizes: if a model has a block-sequential architecture, you can shard it across whatever GPUs you have.

Code + Gradio UI: https://github.com/cronos3k/dreamid-omni-consumer-gpu
Technical writeup: https://huggingface.co/blog/cronos3k/dreamid-omni-sharded

The AI hardware moat is thinner than you think.

#AI #MachineLearning #VideoGeneration #MultiGPU #DreamID #OpenSource

---
