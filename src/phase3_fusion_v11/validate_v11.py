
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DepressionDataset
from models import FusionV11Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, precision_score, recall_score
import argparse
from tqdm import tqdm
from collections import defaultdict
import re

def parse_training_log(log_path):
    """Parses standard output log to extract training history."""
    metrics = []
    # Regex for: Epoch 1: Train Loss=0.9221 | Val AUC=0.5136 | Val F1=0.1333
    pattern = re.compile(r"Epoch (\d+): Train Loss=([\d.]+) \| Val AUC=([\d.]+) \| Val F1=([\d.]+)")
    
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}. Plotting will be skipped.")
        return pd.DataFrame()

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                metrics.append({
                    'epoch': int(match.group(1)),
                    'train_loss': float(match.group(2)),
                    'val_auc': float(match.group(3)),
                    'val_f1': float(match.group(4))
                })
    return pd.DataFrame(metrics)

def plot_loss_auc(df, save_dir):
    if df.empty:
        return
    
    plt.figure(figsize=(10, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # AUC
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_auc'], label='Val AUC', color='green')
    plt.plot(df['epoch'], df['val_f1'], label='Val F1', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_auc_curve.png'))
    plt.close()
    print(f"Saved loss_auc_curve.png to {save_dir}")

def plot_gate_histogram(z_values, save_dir):
    plt.figure(figsize=(8, 6))
    # z values are [N, T, 256] -> flatten. Actually z is often per sample or per step.
    # From model: z is [Batch, T, Shared] or [Batch, Shared] depending on GMU logic.
    # In v11 implementation: z is [Batch, T, Shared].
    # But usually we care about the Average Z per modality (Audio vs Text).
    # Since z is scalar gate for head_a (Audio) vs head_b (Text)?
    # Implementation: `fused_output = z * head_a + (1 - z) * head_b`
    # So z close to 1 => Trust Audio (A), z close to 0 => Trust Text (B)?
    # Wait, check mapping.
    # head_a from audio_query_attn (Query=Audio, Key=Text). This is "Audio-Guided Text"? No.
    # Let's check:
    # head_a, _ = audio_query_attn(query=a_proj, ...) -> Output has shape of Query (Audio). So this is Audio-centric.
    # head_b, _ = text_query_attn(query=t_proj, ...) -> Output has shape of Query (Text). So this is Text-centric.
    # So `z * head_a` emphasizes Audio-centric features.
    # Thus: z -> 1 = Audio Focus. z -> 0 = Text Focus.
    
    z_flat = np.concatenate(z_values).flatten() # Flatten all batches, time steps, dims
    
    sns.histplot(z_flat, bins=50, kde=True, color='purple')
    plt.title("Distribution of Gate Values (z)\n(0 = Text Focus, 1 = Audio Focus)")
    plt.xlabel("Gate Value z")
    plt.ylabel("Frequency")
    plt.axvline(0.5, color='red', linestyle='--')
    plt.text(0.1, plt.ylim()[1]*0.9, "Text Dominant", color='blue')
    plt.text(0.8, plt.ylim()[1]*0.9, "Audio Dominant", color='red')
    
    plt.savefig(os.path.join(save_dir, 'gate_histogram.png'))
    plt.close()
    print(f"Saved gate_histogram.png to {save_dir}")

def find_best_threshold(labels, probs):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_thresh, best_f1

def plot_confusion_matrix(labels, preds, save_dir, threshold):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Depressed', 'Depressed'], yticklabels=['Non-Depressed', 'Depressed'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Thresh={threshold:.3f})')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Saved confusion_matrix.png to {save_dir}")

def validate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset (Use Dev/Validation set for threshold tuning)
    # We need to fit scaler on train first (as usual) to be consistent
    print("Loading Validation Data...")
    train_ds = DepressionDataset(root_dir=args.root_dir, split='train')
    all_scalars = torch.stack(list(train_ds.patient_scalars.values()))
    mean = all_scalars.mean(dim=0)
    std = all_scalars.std(dim=0) + 1e-6
    
    val_dataset = DepressionDataset(root_dir=args.root_dir, split='dev', scalars_scaler=(mean, std))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. Model
    print("Loading Model...")
    model = FusionV11Classifier(dropout=0.0).to(device) # Dropout 0 for inference
    
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        # Fallback to default name in exp dir
        checkpoint_path = os.path.join(args.exp_dir, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    
    # 3. Inference
    all_probs = []
    all_labels = []
    all_z = []
    
    patient_results = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            scalars = batch['scalars'].to(device)
            labels = batch['label'].to(device)
            pids = batch['id']
            
            # Forward
            logits, z = model(audio, text, scalars)
            probs = torch.sigmoid(logits).squeeze(1)
            
            # Collect Z (detached, cpu numpy)
            # z is [B, T, 256]
            all_z.append(z.cpu().numpy())
            
            for i, pid in enumerate(pids):
                patient_results[pid].append({
                    'prob': probs[i].item(),
                    'label': labels[i].item()
                })

    # 4. Aggregation & Metrics
    y_true = []
    y_prob = []
    
    for pid, windows in patient_results.items():
        # Top-3 Aggregation (same as training)
        window_probs = [w['prob'] for w in windows]
        label = windows[0]['label']
        
        top_k = sorted(window_probs, reverse=True)[:3]
        score = sum(top_k) / len(top_k)
        
        y_true.append(label)
        y_prob.append(score)
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # 5. Find Optimal Threshold
    best_thresh, best_f1 = find_best_threshold(y_true, y_prob)
    print(f"\n🏆 Best Threshold: {best_thresh:.4f}")
    print(f"🔥 Max Validation F1: {best_f1:.4f}")
    
    y_pred = (y_prob >= best_thresh).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    print(f"📊 Validation AUC: {auc:.4f}")
    
    # 6. Plots
    os.makedirs(args.save_dir, exist_ok=True)
    
    # - Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, args.save_dir, best_thresh)
    
    # - Gate Histogram
    plot_gate_histogram(all_z, args.save_dir)
    
    # - Loss/AUC Curve (From Logs)
    log_df = parse_training_log(args.log_path)
    plot_loss_auc(log_df, args.save_dir)
    
    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/g755389/DepressiVoice-LLM/', help='Project Root')
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to experiments directory (e.g. experiments/fusion_v11_handover)')
    parser.add_argument('--model_path', type=str, default='', help='Path to .pth file (optional, defaults to exp_dir/best_model.pth)')
    parser.add_argument('--log_path', type=str, default='', help='Path to slurm output log file')
    parser.add_argument('--save_dir', type=str, default='validation_output', help='Directory to save plots')
    
    args = parser.parse_args()
    
    if not args.log_path:
        # Try to find a slurm output in exp_dir
        slurm_files = [f for f in os.listdir(args.exp_dir) if f.startswith('slurm_') and f.endswith('.out')]
        if slurm_files:
            args.log_path = os.path.join(args.exp_dir, slurm_files[0]) # Take first found
            print(f"Auto-detected log file: {args.log_path}")
            
    validate(args)
