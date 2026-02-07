import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import MultimodalDAICDataset, multimodal_collate_fn
from model import FusionModel
# We need to import the SingleModalityModel definition. 
# Since it was defined inside train_baselines.py, we should ideally move it to a shared file or redefine it here.
# For robustness, I will redefine it here to avoid import issues if train_baselines.py changes.
import torch.nn as nn

# --- Model Definition (Must match training script for Pickle) ---
class SingleModalityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask):
        mask_expanded = mask.unsqueeze(-1).float()
        x_masked = x * mask_expanded
        pooled = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        return self.mlp(pooled)

def load_fusion_model(model_path, device):
    # These params should match the V8/V9 config
    model = FusionModel(d_model=256, nhead=8, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_baseline(model_path, modality, device):
    input_dim = 1024 if modality == 'audio' else 768
    model = SingleModalityModel(input_dim=input_dim, hidden_dim=256, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def analyze_synergy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Analyzing Synergy on {device}...")

    # Load Data
    dev_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, 
                                        subset='dev', leakage_json=args.leakage_json)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate_fn)

    # Load Models
    print("Loading Models...")
    fusion_model = load_fusion_model(args.fusion_model_path, device)
    audio_model = load_baseline(args.audio_model_path, 'audio', device)
    text_model = load_baseline(args.text_model_path, 'text', device)

    results = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in dev_loader:
            pid = batch['participant_ids'][0]
            label = batch['label'].item()
            
            # 1. Fusion Pred
            audio = batch['audio_feat'].to(device)
            text = batch['text_feat'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            
            fusion_logits, _ = fusion_model(audio, text, audio_mask, text_mask)
            fusion_prob = torch.sigmoid(fusion_logits).item()
            fusion_pred = 1 if fusion_prob > 0.5 else 0
            
            # 2. Audio Baseline Pred
            audio_logits = audio_model(audio, audio_mask)
            audio_prob = torch.sigmoid(audio_logits).item()
            audio_pred = 1 if audio_prob > 0.5 else 0
            
            # 3. Text Baseline Pred
            text_logits = text_model(text, text_mask)
            text_prob = torch.sigmoid(text_logits).item()
            text_pred = 1 if text_prob > 0.5 else 0
            
            results.append({
                'PID': pid,
                'Label': int(label),
                'Fusion_Pred': fusion_pred,
                'Fusion_Prob': round(fusion_prob, 4),
                'Audio_Pred': audio_pred,
                'Audio_Prob': round(audio_prob, 4),
                'Text_Pred': text_pred,
                'Text_Prob': round(text_prob, 4)
            })

    df = pd.DataFrame(results)
    
    # Identify Synergy Cases:
    # 1. Fusion is Correct
    # 2. At least one modality is Wrong
    synergy_cases = df[
        (df['Fusion_Pred'] == df['Label']) & 
        ((df['Audio_Pred'] != df['Label']) | (df['Text_Pred'] != df['Label']))
    ]
    
    print("\n=== SYNERGY REPORT (Disagreement Analysis) ===")
    print(f"Total Dev Samples: {len(df)}")
    print(f"Fusion Accuracy: {(df['Fusion_Pred'] == df['Label']).mean():.4f}")
    print(f"Audio Accuracy: {(df['Audio_Pred'] == df['Label']).mean():.4f}")
    print(f"Text Accuracy: {(df['Text_Pred'] == df['Label']).mean():.4f}")
    print(f"\nSynergy Cases Found: {len(synergy_cases)}")
    
    if len(synergy_cases) > 0:
        print("\nTop 5 Synergy Examples:")
        print(synergy_cases[['PID', 'Label', 'Fusion_Pred', 'Audio_Pred', 'Text_Pred']].head(5))
        
        # Save detailed report
        out_csv = os.path.join(args.output_dir, "synergy_analysis_report.csv")
        synergy_cases.to_csv(out_csv, index=False)
        print(f"\nSaved detailed analysis to {out_csv}")
    else:
        print("\nNo clear synergy cases found (Fusion didn't correct errors).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized/")
    parser.add_argument("--text_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_clean/")
    parser.add_argument("--labels_csv", type=str, default="/home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv")
    parser.add_argument("--leakage_json", type=str, default="/home/g755389/DepressiVoice-LLM/docs/reports/ellie_leakage_audit.json")
    
    parser.add_argument("--fusion_model_path", type=str, default="/home/g755389/DepressiVoice-LLM/experiments/fusion_v9_seeds/seed_42/best_fusion_model.pt", help="Path to best fusion checkpoint")
    parser.add_argument("--audio_model_path", type=str, default="/home/g755389/DepressiVoice-LLM/experiments/baselines/audio/best_audio_model.pt", help="Path to best audio baseline")
    parser.add_argument("--text_model_path", type=str, default="/home/g755389/DepressiVoice-LLM/experiments/baselines/text/best_text_model.pt", help="Path to best text baseline")
    
    parser.add_argument("--output_dir", type=str, default="/home/g755389/DepressiVoice-LLM/experiments/analysis")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyze_synergy(args)
