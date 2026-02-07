import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_v10 import FusionV10Dataset, v10_collate_fn
from model_v10 import FusionV10Model
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

def train():
    parser = argparse.ArgumentParser(description="Fusion V10 Training")
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--leakage_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_weight", type=float, default=2.5)
    parser.add_argument("--output_dir", type=str, default="experiments/v10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Datasets
    # Pass scalers path to save/load normalization stats
    scalers_path = os.path.join(args.output_dir, "scalars_scaler.pkl")
    
    train_dataset = FusionV10Dataset(
        args.audio_dir, args.text_dir, args.labels_csv, args.leakage_json, 
        subset='train', scalers_path=scalers_path
    )
    dev_dataset = FusionV10Dataset(
        args.audio_dir, args.text_dir, args.labels_csv, args.leakage_json, 
        subset='dev', scalers_path=scalers_path
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=v10_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=v10_collate_fn)

    # Model
    model = FusionV10Model(dropout_transformer=args.dropout, dropout_head=0.35).to(device)
    
    # Loss
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            audio = batch['audio_feat'].to(device)
            text = batch['text_feat'].to(device)
            scalars = batch['scalars'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            # Simple masks for now (Assumes full audio validity for chunks, text padded)
            # v10_collate generates masks
            audio_mask = batch['audio_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(audio, text, scalars, audio_mask, text_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # Validation (Patient-Level Aggregation)
        model.eval()
        dev_loss = 0
        
        # Store window predictions: {pid: [probs]}
        patient_preds = {}
        patient_labels = {}
        
        with torch.no_grad():
            for batch in dev_loader:
                audio = batch['audio_feat'].to(device)
                text = batch['text_feat'].to(device)
                scalars = batch['scalars'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                audio_mask = batch['audio_mask'].to(device)
                text_mask = batch['text_mask'].to(device)
                
                logits, _ = model(audio, text, scalars, audio_mask, text_mask)
                loss = criterion(logits, labels)
                dev_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                lbls = labels.cpu().numpy()
                pids = batch['participant_ids'] # Assuming scalar? No, check collate. list.
                
                for pid, prob, lbl in zip(pids, probs, lbls):
                    if pid not in patient_preds:
                        patient_preds[pid] = []
                        patient_labels[pid] = lbl[0]
                    patient_preds[pid].append(prob[0])
        
        avg_train_loss = train_loss / len(train_loader)
        avg_dev_loss = dev_loss / len(dev_loader)
        
        # Aggregation: Top-3 Mean Pooling
        y_true = []
        y_pred = []
        
        for pid in patient_preds:
            probs = patient_preds[pid]
            # Top-3 Logic
            top_k_probs = sorted(probs, reverse=True)[:3]
            if not top_k_probs:
                score = 0.0
            else:
                score = np.mean(top_k_probs)
            
            final_pred = 1 if score > 0.5 else 0
            y_pred.append(final_pred)
            y_true.append(patient_labels[pid])
            
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        logger.info(f"Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f} | DevLoss={avg_dev_loss:.4f}")
        logger.info(f"Validation: Acc={acc*100:.2f}% | F1={f1:.4f}")
        
        scheduler.step(avg_dev_loss)
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_fusion_v10.pt"))
            logger.info("New Best Model Saved!")

if __name__ == "__main__":
    train()
