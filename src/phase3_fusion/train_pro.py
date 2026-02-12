import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MultimodalDAICDataset, multimodal_collate_fn
from model_pro import FusionModelPro
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        
        if self.pos_weight is not None:
            # Apply pos_weight specifically to samples where target is 1
            weights = torch.ones_like(targets)
            weights[targets == 1] = self.pos_weight
            F_loss = F_loss * weights
            
        return F_loss.mean()

import torch.nn.functional as F

def train():
    parser = argparse.ArgumentParser(description="Phase 3 PRO: Enhanced Fusion Training")
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--leakage_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--pos_weight", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "train.log")),
                                 logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Training PRO Baseline | Device: {device}")

    # Datasets
    train_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, args.leakage_json, subset='train')
    dev_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, args.leakage_json, subset='dev')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_collate_fn)

    # Model
    model = FusionModelPro(dropout=args.dropout).to(device)
    
    # Loss: Focal Loss with Class Weighting
    criterion = FocalLoss(gamma=args.gamma, pos_weight=args.pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    best_auc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            audio, text = batch['audio_feat'].to(device), batch['text_feat'].to(device)
            a_mask, t_mask = batch['audio_mask'].to(device), batch['text_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits, _ = model(audio, text, a_mask, t_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                audio, text = batch['audio_feat'].to(device), batch['text_feat'].to(device)
                a_mask, t_mask = batch['audio_mask'].to(device), batch['text_mask'].to(device)
                logits, _ = model(audio, text, a_mask, t_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.flatten())
                all_labels.extend(batch['label'].numpy())

        auc = roc_auc_score(all_labels, all_probs)
        logger.info(f"Epoch {epoch+1}: Loss: {train_loss/len(train_loader):.4f} | Dev AUC: {auc:.4f}")
        
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_fusion_model_pro.pt"))
            logger.info(f"✨ New Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train()
