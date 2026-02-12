import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Import Phase 3.5 SOTA components
from src.phase3_fusion.model_max import FusionModelMax
import json

def extract_features(checkpoint_path, data_dir, output_dir, device='cuda'):
    """
    Extracts fused embeddings (time-series) and gate values from the frozen Phase 3.5 model.
    """
    
    # 1. Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 2. Load Model
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Initialize v11-PRO FusionModelMax
    model = FusionModelMax(
        d_audio=1024, 
        d_text=768, 
        d_model=256, 
        dropout=0.0
    ).to(device)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle if state_dict was saved as 'model_state_dict' or raw
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # 3. Prepare Patient List
    # We load metadata to get the list of participants and their splits
    try:
        import pandas as pd
        labels_path = os.path.join(data_dir, 'data/01_raw/detailed_lables.csv')
        df = pd.read_csv(labels_path)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Directories
    audio_dir = os.path.join(data_dir, 'data/phase2_clean/audio_embeddings_normalized')
    text_dir = os.path.join(data_dir, 'data/phase2_clean/text_embeddings_clean')

    print(f"Starting extraction for {len(df)} patients...")
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            pid = str(row['Participant'])
            label = int(row['Depression_label'])
            
            # Paths
            audio_path = os.path.join(audio_dir, f"{pid}_features.pt")
            text_path = os.path.join(text_dir, f"{pid}_features.pt")
            
            if not os.path.exists(audio_path) or not os.path.exists(text_path):
                continue
                
            try:
                # Load full tensors
                audio = torch.load(audio_path, map_location=device).float()
                text = torch.load(text_path, map_location=device).float()
                
                # Preprocessing (similar to dataset.py)
                # Handle 3D Audio: [T, Frames, D] -> [T, D]
                if audio.dim() == 3:
                    if audio.shape[1] == 1:
                        audio = audio.squeeze(1)
                    else:
                        audio = audio.mean(dim=1)
                
                # Handle 3D Text
                if text.dim() == 3 and text.shape[1] == 1:
                    text = text.squeeze(1)
                
                # Ensure same length (just in case)
                min_t = min(audio.shape[0], text.shape[0])
                audio = audio[:min_t].unsqueeze(0) # [1, T, 1024]
                text = text[:min_t].unsqueeze(0)   # [1, T, 768]
                
                # Sanitization: Force Audio[t] = 0.0 if Text[t] is "masked" (all zeros)
                is_silence = torch.all(torch.abs(text) < 1e-6, dim=-1) # [1, T]
                audio = audio.clone()
                audio[is_silence] = 0.0
                
                # --- Phenotype-Anchored Extraction Logic (Mirroring model_max.py) ---
                # 1. Projections
                a_feat = torch.relu(model.norm_a(model.audio_proj(audio)))
                t_feat = torch.relu(model.norm_t(model.text_proj(text)))
                
                # 2. Bidirectional Co-Attention
                # Masks: True to MASK
                ma = ~text_mask if 'text_mask' in locals() else None
                h_a, _ = model.audio_query_attn(query=a_feat, key=t_feat, value=t_feat, key_padding_mask=ma)
                
                mt = ~audio_mask if 'audio_mask' in locals() else None
                h_t, _ = model.text_query_attn(query=t_feat, key=a_feat, value=a_feat, key_padding_mask=mt)
                
                # 3. GMU Fusion
                z = model.gmu_gate(torch.cat([h_a, h_t], dim=-1))
                fused = z * h_a + (1 - z) * h_t
                
                # 4. Temporal Context Aggregation (Phase 4 Input)
                fused_seq = model.temporal_head(fused)
                
                # Save
                fused_seq = fused_seq.squeeze(0).cpu()
                gate_vals = z.squeeze(0).cpu()
                
                save_path = os.path.join(output_dir, f"{pid}_fused.pt")
                torch.save({
                    'fused_embedding': fused_seq, # [T, 256] - Contextualized
                    'gate_values': gate_vals,     # [T, 256] - Dynamic Gates
                    'label': label
                }, save_path)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM for {pid}. Sequence length: {audio.shape[1]}")
                    torch.cuda.empty_cache()
                else:
                    print(f"Error extracting {pid}: {e}")
            except Exception as e:
                print(f"Error processing {pid}: {e}")

    print("Extraction Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="experiments/v11_PRO/best_model_max.pt")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="data/phase4_sota_features")
    args = parser.parse_args()
    
    extract_features(args.checkpoint, args.data_root, args.output_dir)
