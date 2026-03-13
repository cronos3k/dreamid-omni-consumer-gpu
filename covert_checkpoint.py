import torch
from safetensors.torch import save_file
from collections import OrderedDict
import os
from tqdm import tqdm



def convert_dreamid_to_ovi(dreamid_path, output_path):
    print(f"Loading DreamID Omni checkpoint from: {dreamid_path}")
    
    # 1. 加载 .pth 文件
    # 注意：通常 EMA 权重存储在 'ema' 或 'state_dict' 键下，或者直接是 state_dict
    # 这里假设 ema.pth 里面包含一个 'ema' key 或者直接是权重字典
    checkpoint = torch.load(dreamid_path, map_location="cpu")
    
    # if "ema" in checkpoint:
    #     state_dict = checkpoint["ema"]
    #     print("Loaded weights from 'ema' key.")
    # elif "state_dict" in checkpoint:
    #     state_dict = checkpoint["state_dict"]
    #     print("Loaded weights from 'state_dict' key.")
    # else:
    state_dict = checkpoint
    print("Loaded weights directly.")

    new_state_dict = OrderedDict()
    
    print("Starting conversion...")
    
 
    for old_key, value in tqdm(state_dict.items(), desc="Converting keys"):
        new_key = old_key
        

        if old_key.startswith('blocks.') and '.vid_block.' in old_key:
            parts = old_key.split('.')
       
            block_idx = parts[1]
            rest = '.'.join(parts[3:])
            new_key = f"video_model.blocks.{block_idx}.{rest}"
            

        elif old_key.startswith('blocks.') and '.audio_block.' in old_key:
            parts = old_key.split('.')
   
            block_idx = parts[1]
            rest = '.'.join(parts[3:])
            new_key = f"audio_model.blocks.{block_idx}.{rest}"
            

        if 'modulation' in new_key and 'head' not in new_key:
          
            new_key = new_key.replace('.modulation', '.modulation.modulation')

        new_state_dict[new_key] = value

    print(f"Converted {len(state_dict)} keys to {len(new_state_dict)} keys.")
    
    # 3. 保存为 .safetensors
    print(f"Saving to {output_path}...")
    save_file(new_state_dict, output_path)
    print("Done!")

# --- 配置路径 ---
dreamid_omni_path = '/mnt/bn/mouch-hl-1/dreamid_omni_guoxu/ckpts/dreamid_omni_all_part4_new_rope/7500/ema.pth'
output_ovi_path = '/mnt/bn/mouch-hl-1/dreamid_omni_nvidia/ckpts/DreamID_Omni/dreamid_omni.safetensors'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_ovi_path), exist_ok=True)

# 执行转换
convert_dreamid_to_ovi(dreamid_omni_path, output_ovi_path)
