import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import argparse
from tqdm import tqdm

from .dataset import TemporalDataset, pad_collate
from .models import CrossModalTransformerAggregator, FocalLoss

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    all_preds_reg, all_targets_reg = [], []
    
    with torch.no_grad():
        for batch in loader:
            features, mask = batch['features'].to(device), batch['mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            target_reg = batch['phq_scores'].cpu().numpy()
            
            # Unpack MTL output
            (logits, severity), _ = model(features, mask)
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            reg_out = severity.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_preds_reg.extend(reg_out)
            all_targets_reg.extend(target_reg)
            
    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.5
    
    # Calculate RMSE for Severity
    rmse = np.sqrt(np.mean((np.array(all_preds_reg) - np.array(all_targets_reg))**2))
    
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    return {'auc': auc, 'f1': f1, 'rmse': rmse, 'probs': all_probs, 'labels': all_labels}

def train_phase4(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on {device}")
    
    # Dataset
    train_ds = TemporalDataset(args.data_dir, args.labels_csv, augment=True)
    dev_ds = TemporalDataset(args.data_dir, args.labels_csv, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    
    # Model: CrossModalTransformer v7 (MTL)
    model = CrossModalTransformerAggregator(input_dim=256, d_model=args.d_model, nhead=args.nhead).to(device)
    
    # Loss Functions
    criterion_cls = FocalLoss(alpha=0.75, gamma=2.0) 
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_auc = 0.0
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)
            phq_target = batch['phq_scores'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            (logits, severity), _ = model(features, mask)
            
            loss_cls = criterion_cls(logits, labels)
            loss_reg = criterion_reg(severity, phq_target)
            
            # Joint Loss: Focal + 0.5 * MSE
            loss = loss_cls + (0.5 * loss_reg)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        metrics = evaluate(model, dev_loader, device)
        print(f"E{epoch+1} | Loss: {total_loss/len(train_loader):.4f} | AUC: {metrics['auc']:.4f} | RMSE: {metrics['rmse']:.4f}")
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_mtl_transformer.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_phase4(args)
