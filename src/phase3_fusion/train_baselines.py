import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MultimodalDAICDataset, multimodal_collate_fn
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

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
        # x: [B, T, D]
        # mask: [B, T]
        
        # Mean pooling over non-masked indices
        mask_expanded = mask.unsqueeze(-1).float()
        x_masked = x * mask_expanded
        pooled = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        
        return self.mlp(pooled)

def train_baseline(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {args.modality.upper()} Baseline on {device}")

    # Datasets
    train_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, 
                                          subset='train', leakage_json=args.leakage_json)
    dev_dataset = MultimodalDAICDataset(args.audio_dir, args.text_dir, args.labels_csv, 
                                        subset='dev', leakage_json=args.leakage_json)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_collate_fn)

    input_dim = 1024 if args.modality == 'audio' else 768
    model = SingleModalityModel(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = 0
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            x = batch[f'{args.modality}_feat'].to(device)
            mask = batch[f'{args.modality}_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dev_loader:
                x = batch[f'{args.modality}_feat'].to(device)
                mask = batch[f'{args.modality}_mask'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                
                logits = model(x, mask)
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        f1 = f1_score(all_labels, all_preds)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_{args.modality}_model.pt"))
            logger.info(f"Epoch {epoch+1}: New Best F1: {f1:.4f}")

    logger.info(f"Training Complete. Best {args.modality.upper()} F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, required=True, choices=['audio', 'text'])
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--leakage_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="baselines")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--pos_weight", type=float, default=3.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"Baseline_{args.modality}")
    
    train_baseline(args, logger)
