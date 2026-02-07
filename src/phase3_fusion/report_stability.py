import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from dataset import MultimodalDAICDataset, multimodal_collate_fn
from model import FusionModel

def evaluate_model(model_path, dev_loader, device):
    if not os.path.exists(model_path):
        return None, None
        
    # Load Model
    model = FusionModel(d_model=256, nhead=8, num_layers=2, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dev_loader:
            audio = batch['audio_feat'].to(device)
            text = batch['text_feat'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(audio, text, audio_mask, text_mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    return f1, acc

def report_stability(args):
    print("=== Fusion V9 Stability Report (Offline Evaluation) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data Once
    print("Loading Dev Set...")
    dev_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, 
                                        leakage_json=args.leakage_json, subset='dev')
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=multimodal_collate_fn)
    
    # Find Models
    seed_dirs = glob.glob(os.path.join(args.experiments_dir, "seed_*"))
    data = []
    
    print(f"Found {len(seed_dirs)} seed directories in {args.experiments_dir}")
    
    for d in seed_dirs:
        seed = os.path.basename(d).split('_')[-1]
        model_path = os.path.join(d, "best_fusion_model.pt")
        
        print(f"Evaluating Seed {seed}...")
        try:
            f1, acc = evaluate_model(model_path, dev_loader, device)
            
            if f1 is not None:
                data.append({
                    'Seed': seed,
                    'F1': f1,
                    'Accuracy': acc
                })
                print(f"  -> F1: {f1:.4f} | Acc: {acc:.4f}")
            else:
                print(f"  -> Model not found at {model_path}")
        except Exception as e:
            print(f"  -> Error evaluating seed {seed}: {e}")
            
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("No valid results found.")
        return

    print("\n--- Individual Runs ---")
    print(df.sort_values('Seed'))
    
    print("\n--- Thesis Statistics ---")
    mean_f1 = df['F1'].mean()
    std_f1 = df['F1'].std()
    mean_acc = df['Accuracy'].mean()
    std_acc = df['Accuracy'].std()
    
    print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    # Thesis Table Format
    print("\n[LaTeX Format]")
    print(f"DepressiVoice (Fusion) & {mean_acc*100:.2f} $\pm$ {std_acc*100:.2f} & {mean_f1:.4f} $\pm$ {std_f1:.4f} \\\\")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", type=str, default="/home/g755389/DepressiVoice-LLM/experiments/fusion_v9_seeds")
    
    # Data Paths
    parser.add_argument("--audio_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized/")
    parser.add_argument("--text_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/phase2_clean/text_embeddings_clean/")
    parser.add_argument("--labels_csv", type=str, default="/home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv")
    parser.add_argument("--leakage_json", type=str, default="/home/g755389/DepressiVoice-LLM/docs/reports/ellie_leakage_audit.json")
    
    args = parser.parse_args()
    
    report_stability(args)
