import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
import argparse

from src.phase4_sota.dataset import TemporalDataset, pad_collate
from src.phase4_sota.models import ImprovedLSTMAggregator, CrossModalTransformerAggregator

def optimize(data_dir, model_path, output_plot="experiments/phase4_temporal/pr_curve.png", device='cuda', model_type='transformer'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    # 1. Load Data (Dev Set)
    try:
        labels_path = os.path.join("data", "01_raw", "detailed_labels.csv")
        if not os.path.exists(labels_path):
            labels_path = "data/01_raw/detailed_lables.csv"
        df = pd.read_csv(labels_path)
        dev_ids = df[df['split']=='dev']['Participant'].astype(str).tolist()
    except:
        print("⚠️ Could not find labels for dev split. Using all files.")
        dev_ids = None

    dataset = TemporalDataset(data_dir, split_ids=dev_ids, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)

    # 2. Load Model
    if model_type == 'lstm':
        model = ImprovedLSTMAggregator(input_dim=256, hidden_dim=64, dropout=0.0).to(device)
    else:
        model = CrossModalTransformerAggregator(
            input_dim=256,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.0
        ).to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            (logits, severity), _ = model(features, mask)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 3. Exhaustive Search
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = 0
    best_thresh = 0.5
    
    results = []
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        
        results.append({'threshold': t, 'f1': f1, 'precision': prec, 'recall': rec})
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    res_df = pd.DataFrame(results)
    print(f"\n🏆 Global Maximum F1: {best_f1:.4f} at Threshold: {best_thresh:.4f}")
    
    # Detailed final report at best threshold
    best_preds = (all_probs >= best_thresh).astype(int)
    print(f"   -> Precision: {precision_score(all_labels, best_preds):.4f}")
    print(f"   -> Recall:    {recall_score(all_labels, best_preds):.4f}")
    print(f"   -> AUC:       {roc_auc_score(all_labels, all_probs):.4f}")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['threshold'], res_df['f1'], label='F1-Score', color='blue', lw=2)
    plt.plot(res_df['threshold'], res_df['precision'], label='Precision', linestyle='--', alpha=0.7)
    plt.plot(res_df['threshold'], res_df['recall'], label='Recall', linestyle='--', alpha=0.7)
    plt.axvline(best_thresh, color='red', linestyle=':', label=f'Best Thresh: {best_thresh:.2f}')
    
    plt.title("Phase 4: F1-Score Optimization Curve")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot)
    print(f"✅ Optimization plot saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/phase4_sota_features")
    parser.add_argument("--model_path", type=str, default="experiments/phase4_v11_PRO_SOTA/best_mtl_transformer.pth")
    parser.add_argument("--output_plot", type=str, default="experiments/phase4_v11_PRO_SOTA/f1_optimization_sota.png")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["lstm", "transformer"])
    args = parser.parse_args()
    
    optimize(args.data_dir, args.model_path, args.output_plot, model_type=args.model_type)
