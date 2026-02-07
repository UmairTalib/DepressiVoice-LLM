import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DepressionDataset
from models import FusionV11Classifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
POS_WEIGHT = 2.5
LABEL_SMOOTHING = 0.1
PROJECT_ROOT = "/home/g755389/DepressiVoice-LLM/" 

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Datasets
    # First, init train dataset to fit scaler
    train_dataset = DepressionDataset(root_dir=args.root_dir, split='train')
    
    # Calculate scalar statistics for normalization
    all_scalars = torch.stack(list(train_dataset.patient_scalars.values()))
    scalar_mean = all_scalars.mean(dim=0)
    scalar_std = all_scalars.std(dim=0) + 1e-6 
    
    # helper for normalization
    def normalize_scalars(dataset, mean, std):
        for pid in dataset.patient_scalars:
            dataset.patient_scalars[pid] = (dataset.patient_scalars[pid] - mean) / std
            
    normalize_scalars(train_dataset, scalar_mean, scalar_std)
    
    dev_dataset = DepressionDataset(root_dir=args.root_dir, split='dev')
    normalize_scalars(dev_dataset, scalar_mean, scalar_std) # Use TRAIN stats
    
    test_dataset = DepressionDataset(root_dir=args.root_dir, split='test')
    normalize_scalars(test_dataset, scalar_mean, scalar_std) # Use TRAIN stats
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model - v11 Classifier
    print("Initializing FusionV11Classifier...")
    model = FusionV11Classifier(
        audio_dim=1024,
        text_dim=768,
        shared_dim=256,
        nhead=8,
        dropout=0.4,
        num_scalars=3
    ).to(device)
    
    # 3. Optimization
    class_weights = torch.tensor([POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # 4. Training Loop
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in loop:
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            scalars = batch['scalars'].to(device)
            labels = batch['label'].to(device).unsqueeze(1) # [B, 1]
            
            # Label Smoothing
            smooth_labels = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            
            optimizer.zero_grad()
            # Forward pass returns (logits, z)
            logits, z = model(audio, text, scalars)
            loss = criterion(logits, smooth_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val AUC={val_metrics['auc']:.4f} | Val F1={val_metrics['f1']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print("  New best model saved!")

def evaluate(model, loader, device):
    model.eval()
    results = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Eval]"):
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            scalars = batch['scalars'].to(device)
            labels = batch['label'].to(device) # [B]
            pids = batch['id'] # tuple of IDs
            
            logits, z = model(audio, text, scalars)
            probs = torch.sigmoid(logits).squeeze(1) # [B]
            
            for i, pid in enumerate(pids):
                results[pid].append({
                    'prob': probs[i].item(),
                    'label': labels[i].item()
                })
    
    # Patient-level Aggregation: Top-K Mean Pooling (K=3)
    patient_preds = []
    patient_labels = []
    
    for pid, windows in results.items():
        # windows is list of dicts
        window_probs = [w['prob'] for w in windows]
        label = windows[0]['label'] # All windows have same label
        
        # Top-K (K=3)
        # If fewer than K windows, take all.
        top_k = sorted(window_probs, reverse=True)[:3]
        score = sum(top_k) / len(top_k)
        
        patient_preds.append(score)
        patient_labels.append(label)
        
    # Metrics
    try:
        auc = roc_auc_score(patient_labels, patient_preds)
    except:
        auc = 0.5 # Handle case with only one class
        
    preds_binary = [1 if p > 0.5 else 0 for p in patient_preds]
    f1 = f1_score(patient_labels, preds_binary, zero_division=0)
    
    return {'auc': auc, 'f1': f1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/Users/umairtalib/DepressiVoice-LLM/', help='Project root directory')
    parser.add_argument('--save_dir', type=str, default='./enc_output', help='Directory to save models')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
