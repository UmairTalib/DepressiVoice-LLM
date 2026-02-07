import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from dataset import MultimodalDAICDataset, multimodal_collate_fn
from model import FusionModel


def visualize_attention(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Visualizing Attention for Session {args.session_id} on {device}")
    
    # Load Data (Single Session)
    dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, 
                                       leakage_json=args.leakage_json, subset='dev')
    
    # Filter for specific session
    target_sample = None
    for s in dataset.samples:
        if str(s['participant_id']) == str(args.session_id):
            target_sample = s
            break
            
    if target_sample is None:
        print(f"Session {args.session_id} not found in Dev set.")
        return

    # Manually collate
    batch = multimodal_collate_fn([dataset.__getitem__(dataset.samples.index(target_sample))])
    
    # Load Model
    model = FusionModel(d_model=256, nhead=8, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Inference
    with torch.no_grad():
        audio = batch['audio_feat'].to(device)
        text = batch['text_feat'].to(device)
        audio_mask = batch['audio_mask'].to(device)
        text_mask = batch['text_mask'].to(device)
        
        logits, attn_weights = model(audio, text, audio_mask, text_mask)
        prob = torch.sigmoid(logits).item()
        
    print(f"Prediction: {prob:.4f} (Label: {target_sample['label']})")
    
    # Shape: [Batch, T_text, T_audio] (Already averaged across heads by default)
    # We take the first sample -> [T_text, T_audio]
    attn = attn_weights[0].cpu().numpy()
    
    # Remove padding from visual
    valid_text = text_mask[0].sum().item()
    valid_audio = audio_mask[0].sum().item()
    attn = attn[:valid_text, :valid_audio]
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.heatmap(attn, cmap='viridis', xticklabels=5, yticklabels=5)
    plt.xlabel("Audio Windows (8s)")
    plt.ylabel("Text Windows (8s)")
    plt.title(f"Cross-Modal Attention: Session {args.session_id} (Pred: {prob:.2f})")
    
    out_path = os.path.join(args.output_dir, f"attention_{args.session_id}.png")
    plt.savefig(out_path)
    print(f"Saved attention map to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized/")
    parser.add_argument("--text_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_clean/")
    parser.add_argument("--labels_csv", type=str, default="/home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv")
    parser.add_argument("--leakage_json", type=str, default="/home/g755389/DepressiVoice-LLM/docs/reports/ellie_leakage_audit.json")
    parser.add_argument("--output_dir", type=str, default="/home/g755389/DepressiVoice-LLM/experiments/analysis")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    visualize_attention(args)
