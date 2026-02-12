import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

# Local imports
from dataset import DepressionDataset
from models import FusionV10Model

def evaluate_on_set(model, loader, device, threshold=0.5):
    """
    Standard evaluation for a given split and threshold.
    Returns patient-level metrics.
    """
    model.eval()
    results = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Inference]", leave=False):
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            scalars = batch['scalars'].to(device)
            labels = batch['label'].to(device)
            pids = batch['id']
            
            logits = model(audio, text, scalars)
            probs = torch.sigmoid(logits).squeeze(1)
            
            for i, pid in enumerate(pids):
                results[pid].append({
                    'prob': probs[i].item(),
                    'label': labels[i].item()
                })
    
    # Patient-level Aggregation: Top-K Mean Pooling (K=3)
    patient_probs = []
    patient_labels = []
    
    for pid, windows in results.items():
        window_probs = [w['prob'] for w in windows]
        label = windows[0]['label']
        
        # Top-K (K=3)
        top_k = sorted(window_probs, reverse=True)[:3]
        score = sum(top_k) / len(top_k)
        
        patient_probs.append(score)
        patient_labels.append(label)
        
    preds_binary = [1 if p >= threshold else 0 for p in patient_probs]
    
    # Calculate Metrics
    try:
        auc = roc_auc_score(patient_labels, patient_probs)
    except:
        auc = 0.5
        
    return {
        'auc': auc,
        'f1': f1_score(patient_labels, preds_binary, zero_division=0),
        'precision': precision_score(patient_labels, preds_binary, zero_division=0),
        'recall': recall_score(patient_labels, preds_binary, zero_division=0),
        'accuracy': accuracy_score(patient_labels, preds_binary),
        'cm': confusion_matrix(patient_labels, preds_binary),
        'probs': patient_probs,
        'labels': patient_labels
    }

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Datasets and Handle Normalization
    print("\nFitting Scaler on Train set...")
    train_dataset = DepressionDataset(root_dir=args.root_dir, split='train')
    train_scaler = train_dataset.scalars_scaler
    
    print("\nLoading Dev & Test sets...")
    dev_dataset = DepressionDataset(root_dir=args.root_dir, split='dev', scalars_scaler=train_scaler)
    test_dataset = DepressionDataset(root_dir=args.root_dir, split='test', scalars_scaler=train_scaler)
    
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 2. Load Model
    print(f"\nLoading Model from {args.model_path}...")
    model = FusionV10Model(hidden_dim=256, num_layers=2).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 3. Threshold Tuning on Dev Set
    print("\nStep 1: Threshold Tuning on DEV Set...")
    # Get probabilities once to speed up tuning
    dev_results = evaluate_on_set(model, dev_loader, device, threshold=0.5)
    patient_probs = dev_results['probs']
    patient_labels = dev_results['labels']
    
    best_f1 = -1
    best_threshold = 0.5
    
    thresholds = np.arange(0.2, 0.61, 0.01)
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in patient_probs]
        f1 = f1_score(patient_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    print(f"Optimal Threshold Found: {best_threshold:.2f} (Dev F1: {best_f1:.4f})")
    print(f"Dev AUC (at this threshold): {dev_results['auc']:.4f}")
    
    # 4. Official Evaluation on Test Set
    print("\nStep 2: Official Testing on TEST Set...")
    test_results = evaluate_on_set(model, test_loader, device, threshold=best_threshold)
    
    print("-" * 30)
    print("FINAL TEST METRICS")
    print("-" * 30)
    print(f"AUC:       {test_results['auc']:.4f}")
    print(f"F1 Score:  {test_results['f1']:.4f}")
    print(f"Accuracy:  {test_results['accuracy']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall:    {test_results['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(test_results['cm'])
    
    # Determine Class Distribution for context
    pos = sum(test_results['labels'])
    total = len(test_results['labels'])
    print(f"\nTest Distribution: {int(pos)} Depressed / {total} Total ({(pos/total)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/g755389/DepressiVoice-LLM/')
    parser.add_argument('--model_path', type=str, default='/home/g755389/DepressiVoice-LLM/models/fusion_v10/best_model.pth')
    
    args = parser.parse_args()
    main(args)
