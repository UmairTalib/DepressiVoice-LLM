
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DepressionAudioDataset
from audiomodel import DepressionAudioModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # Dataset
    train_dataset = DepressionAudioDataset(args.data_dir, args.labels_csv, subset='train')
    # val_dataset = DepressionAudioDataset(args.data_dir, args.labels_csv, subset='dev') # If we had splits
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = DepressionAudioModel(num_classes=2)
    model.to(device)
    
    # Optimizer & Loss
    # Use calculated weight from dataset if available
    if hasattr(train_dataset, 'pos_weight'):
        # Pass weights to CrossEntropyLoss [weight_class_0, weight_class_1]
        # class 0 is usually majority (weight 1.0), class 1 is minority (weight > 1.0)
        class_weights = torch.tensor([1.0, train_dataset.pos_weight]).to(device)
        print(f"Using Weighted Loss: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Mixed Precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training Loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_values, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                
                # Backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_values, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            
            # Metrics
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Macro-F1: {epoch_f1:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_interval", type=int, default=1)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
