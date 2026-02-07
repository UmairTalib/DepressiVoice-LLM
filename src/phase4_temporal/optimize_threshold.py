import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
import argparse

from .dataset import TemporalDataset, pad_collate
from .models import CrossModalTransformerAggregator

def optimize(model, loader, device, output_plot):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            features, mask = batch['features'].to(device), batch['mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Unpack MTL output
            (logits, severity), _ = model(features, mask)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(all_labels, (all_probs >= t).astype(int), zero_division=0) for t in thresholds]
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='blue', linewidth=2)
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.title(f'F1 Optimization (Peak F1: {best_f1:.4f})')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_plot, dpi=300)
    print(f"📊 F1 Curve saved to {output_plot}")
    
    return best_threshold, best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--output_plot", type=str, default="f1_optimization_v7_mtl.png")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrossModalTransformerAggregator(input_dim=256).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    ds = TemporalDataset(args.data_dir, args.labels_csv, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=pad_collate)
    
    t, f = optimize(model, loader, device, args.output_plot)
    print(f"✅ Optimal Threshold: {t:.4f} | Max F1: {f:.4f}")
