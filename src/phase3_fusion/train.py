import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MultimodalDAICDataset, multimodal_collate_fn
from model import FusionModel
import logging
from tqdm import tqdm

def train():
    parser = argparse.ArgumentParser(description="Phase 3: Fusion Network Training")
    parser.add_argument("--audio_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/03_features/audio_embeddings_clean/")
    parser.add_argument("--text_dir", type=str, default="/home/g755389/DepressiVoice-LLM/data/03_features/text_embeddings_seq/")
    parser.add_argument("--labels_csv", type=str, default="/home/g755389/DepressiVoice-LLM/data/01_raw/detailed_lables.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_audio_len", type=int, default=5000)
    parser.add_argument("--max_text_len", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="checkpoints/fusion_v1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Datasets & Loaders
    train_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, subset='train', max_audio_len=args.max_audio_len, max_text_len=args.max_text_len)
    dev_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, subset='dev', max_audio_len=args.max_audio_len, max_text_len=args.max_text_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_collate_fn)

    # Model
    model = FusionModel(d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dropout=args.dropout).to(device)
    
    # Loss Strategy: Weighted BCE
    # Imbalance 1:3.17 (24% Depressed)
    pos_weight = torch.tensor([3.17]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_dev_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            audio = batch['audio_feat'].to(device)
            text = batch['text_feat'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1) # [B, 1]
            
            optimizer.zero_grad()
            logits, _ = model(audio, text, audio_mask, text_mask)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        dev_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dev_loader:
                audio = batch['audio_feat'].to(device)
                text = batch['text_feat'].to(device)
                audio_mask = batch['audio_mask'].to(device)
                text_mask = batch['text_mask'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                
                logits, _ = model(audio, text, audio_mask, text_mask)
                loss = criterion(logits, labels)
                dev_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_dev_loss = dev_loss / len(dev_loader)
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | Acc: {accuracy*.100:.2f}%")
        
        scheduler.step(avg_dev_loss)
        
        # Save Best
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_fusion_model.pt"))
            logger.info(f"New best model saved at epoch {epoch+1}")

if __name__ == "__main__":
    train()
