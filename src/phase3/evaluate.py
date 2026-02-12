import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from collections import defaultdict

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.phase3_fusion_v11.dataset import DepressionDataset
from src.phase3_fusion.model_max import FusionModelMax

def evaluate_pro(root_dir, model_path, labels_csv=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Evaluating v11-PRO on: {device}")

    # 1. Datasets
    train_dataset = DepressionDataset(root_dir=root_dir, split='train', labels_csv=labels_csv)
    dev_dataset = DepressionDataset(root_dir=root_dir, split='dev', labels_csv=labels_csv, 
                                    scalars_scaler=train_dataset.scalars_scaler)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # 2. Model
    model = FusionModelMax().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = defaultdict(list)

    # 3. Inference
    with torch.no_grad():
        for batch in dev_loader:
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            scalars = batch['scalars'].to(device)
            labels = batch['label'].to(device).cpu().numpy()
            pids = batch['id']

            out = model(audio, text, scalars)
            probs = torch.sigmoid(out['logits']).squeeze(1).cpu().numpy()

            for i, pid in enumerate(pids):
                results[pid].append({
                    'prob': probs[i],
                    'label': labels[i]
                })

    # 4. Aggregation & Metrics
    y_true, y_score = [], []
    for pid, windows in results.items():
        window_probs = [w['prob'] for w in windows]
        label = windows[0]['label']
        # Top-3 mean aggregation
        score = np.mean(sorted(window_probs, reverse=True)[:3])
        y_true.append(label)
        y_score.append(score)

    y_pred = (np.array(y_score) > 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)

    print("\n" + "="*30)
    print("FINAL v11-PRO RESULTS")
    print("="*30)
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("="*30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, default=None)
    args = parser.parse_args()
    
    evaluate_pro(args.root_dir, args.model_path, args.labels_csv)
