import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd

# Import your model and dataset
from src.phase4_sota.dataset import TemporalDataset, pad_collate
from src.phase4_sota.models import ImprovedLSTMAggregator, CrossModalTransformerAggregator, FocalLoss

def train_phase4(
    data_dir,
    output_dir,
    batch_size=8,
    epochs=50,
    lr=1e-4,
    device='cuda',
    model_type='transformer'
):
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training Phase 4 ({model_type.upper()}) on {device}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Automatic Train/Dev Split
    try:
        labels_path = "data/labels/master_labels_v7.csv"
        df = pd.read_csv(labels_path)
        df['Participant_ID'] = df['Participant_ID'].astype(str)
        train_ids = df[df['split']=='train']['Participant_ID'].tolist()
        dev_ids = df[df['split']=='dev']['Participant_ID'].tolist()
        print(f"✅ Loaded Splits from v7: {len(train_ids)} Train, {len(dev_ids)} Dev")
    except Exception as e:
        print(f"⚠️ Metadata not found. Using random split. Error: {e}")
        all_ids = list(set([f.split('_')[0] for f in os.listdir(data_dir) if f.endswith('.pt')]))
        train_ids, dev_ids = all_ids[:int(0.8*len(all_ids))], all_ids[int(0.8*len(all_ids)):]
    
    # 2. Datasets
    train_ds = TemporalDataset(data_dir, split_ids=train_ids, augment=True)
    dev_ds = TemporalDataset(data_dir, split_ids=dev_ids, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    
    # 3. Initialize Model
    if model_type == 'lstm':
        model = ImprovedLSTMAggregator(input_dim=256, hidden_dim=64, num_layers=1, dropout=0.5, temperature=1.0).to(device)
        model_name = "best_mtl_lstm.pth"
    else:
        model = CrossModalTransformerAggregator(input_dim=256, d_model=128, nhead=4, num_layers=2, dropout=0.3, temperature=1.0).to(device)
        model_name = "best_mtl_transformer.pth"
    
    # Loss Functions
    criterion_cls = FocalLoss(alpha=0.75, gamma=2.0) 
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_metric = 0.0
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
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
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        metrics = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val AUC: {metrics['auc']:.4f} | RMSE: {metrics['rmse']:.4f}")
        
        # Save based on AUC (overall depression detection quality)
        if metrics['auc'] > best_metric:
            best_metric = metrics['auc']
            torch.save(model.state_dict(), os.path.join(output_dir, model_name))
            
    print("\n🏆 Training Complete.")
    print("------------------------------------------------")
    
    # 5. Final Optimization
    print(f"🔎 Searching for Optimal Threshold for {model_type.upper()}...")
    model.load_state_dict(torch.load(os.path.join(output_dir, model_name)))
    optimize_threshold(model, dev_loader, device)

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    all_preds_reg, all_targets_reg = [], []
    
    with torch.no_grad():
        for batch in loader:
            features, mask = batch['features'].to(device), batch['mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            target_reg = batch['phq_scores'].cpu().numpy()
            
            (logits, severity), _ = model(features, mask)
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            reg_out = severity.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_preds_reg.extend(reg_out)
            all_targets_reg.extend(target_reg)
            
    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.5
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(all_preds_reg) - np.array(all_targets_reg))**2))
    
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    return {'auc': auc, 'f1': f1, 'rmse': rmse, 'probs': all_probs, 'labels': all_labels}

def optimize_threshold(model, loader, device):
    metrics = evaluate(model, loader, device)
    probs, labels = np.array(metrics['probs']), np.array(metrics['labels'])
    best_f1, best_thresh, best_recall = 0, 0.5, 0
    thresholds = np.arange(0.01, 0.5, 0.01) 
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh, best_recall = f1, t, rec
            
    print(f"✅ Optimal Threshold Found: {best_thresh:.4f}")
    print(f"   -> F1 Score: {best_f1:.4f}")
    print(f"   -> Recall:   {best_recall:.4f}")
    print(f"   -> Val RMSE: {metrics['rmse']:.4f}")
    
    final_preds = (probs >= best_thresh).astype(int)
    print(f"   -> Precision: {precision_score(labels, final_preds, zero_division=0):.4f}")
    print(f"   -> AUC:      {metrics['auc']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/phase4_sota_features")
    parser.add_argument("--output_dir", type=str, default="experiments/phase4_v11_PRO_SOTA")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_type", type=str, default="transformer", choices=["lstm", "transformer"])
    args = parser.parse_args()
    train_phase4(args.data_dir, args.output_dir, epochs=args.epochs, lr=args.lr, model_type=args.model_type)
