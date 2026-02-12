import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import logging
from collections import defaultdict

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.phase3_fusion_v11.dataset import DepressionDataset
from src.phase3_fusion.model_max import FusionModelMax

def orthogonal_loss(h_a, h_t):
    """
    Forces the text and audio streams to be decorrelated.
    """
    # h: [B, T, D]
    # Flatten across batch and time
    a = h_a.reshape(-1, h_a.shape[-1])
    t = h_t.reshape(-1, h_t.shape[-1])
    
    # Cosine similarity matrix
    a_norm = F.normalize(a, p=2, dim=1)
    t_norm = F.normalize(t, p=2, dim=1)
    
    # Element-wise cosine similarity squared
    similarity = (a_norm * t_norm).sum(dim=1)**2
    return similarity.mean()

def train():
    parser = argparse.ArgumentParser(description="Phase 3.5 MAX: State-of-the-Art Fusion")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pos_weight", type=float, default=3.2)
    parser.add_argument("--ortho_lambda", type=float, default=0.1)
    parser.add_argument("--mse_lambda", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "train_max.log")),
                                 logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"👑 Training Phase 3.5 MAX | Device: {device}")

    # Datasets
    # First, init train to get scaler
    train_dataset = DepressionDataset(root_dir=args.root_dir, split='train', labels_csv=args.labels_csv)
    # Scaler is already fitted in __init__ for train subset in the v11 dataset.py
    
    dev_dataset = DepressionDataset(root_dir=args.root_dir, split='dev', labels_csv=args.labels_csv, 
                                    scalars_scaler=train_dataset.scalars_scaler)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = FusionModelMax().to(device)
    
    # Loss
    class_weights = torch.tensor([args.pos_weight]).to(device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_bce = 0
        epoch_ortho = 0
        epoch_mse = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            scalars = batch['scalars'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass returns dict
            out = model(audio, text, scalars)
            logits = out['logits']
            
            # Loss 1: Task Loss
            loss_bce = bce_criterion(logits, labels)
            
            # Loss 2: Orthogonal Constraint
            loss_ortho = orthogonal_loss(out['h_audio'], out['h_text'])
            
            # Loss 3: PHQ-8 Phenotype Regression (Severity + 8 Symptoms)
            loss_mse_sev = F.mse_loss(out['severity'], batch['severity'].to(device).unsqueeze(1))
            loss_mse_sym = F.mse_loss(out['symptoms'], batch['symptoms'].to(device))
            loss_mse = loss_mse_sev + loss_mse_sym
            
            total_loss = loss_bce + (args.ortho_lambda * loss_ortho) + (args.mse_lambda * loss_mse)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_bce += loss_bce.item()
            epoch_ortho += loss_ortho.item()
            epoch_mse += loss_mse.item()

        scheduler.step()

        # Validation (Patient-level)
        model.eval()
        results = defaultdict(list)
        with torch.no_grad():
            for batch in dev_loader:
                audio = batch['audio'].to(device)
                text = batch['text'].to(device)
                scalars = batch['scalars'].to(device)
                labels = batch['label'].to(device)
                pids = batch['id']
                
                out = model(audio, text, scalars)
                probs = torch.sigmoid(out['logits']).squeeze(1).cpu().numpy()
                
                for i, pid in enumerate(pids):
                    results[pid].append({
                        'prob': probs[i],
                        'label': labels[i].item()
                    })

        # Aggregation Logic (Top-3 Mean)
        y_true, y_score = [], []
        for pid, windows in results.items():
            window_probs = [w['prob'] for w in windows]
            label = windows[0]['label']
            score = np.mean(sorted(window_probs, reverse=True)[:3])
            y_true.append(label)
            y_score.append(score)

        auc = roc_auc_score(y_true, y_score)
        logger.info(f"Epoch {epoch+1}: Loss: {epoch_loss/len(train_loader):.4f} | BCE: {epoch_bce/len(train_loader):.3f} | Ortho: {epoch_ortho/len(train_loader):.3f} | MSE: {epoch_mse/len(train_loader):.3f} | Dev AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_max.pt"))
            logger.info(f"🌟 SOTA Achieved! New Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train()
