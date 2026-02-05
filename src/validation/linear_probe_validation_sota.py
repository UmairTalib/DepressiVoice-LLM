"""
SOTA Linear Probe Validation
Validates WavLM-Large (1024-dim) features against detailed_lables.csv.
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

def validate_sota(features_dir, labels_csv):
    print(f"Loading labels from {labels_csv}...")
    df = pd.read_csv(labels_csv)
    # Using 'Participant' and 'Depression_label' as provided by user
    labels_dict = dict(zip(df['Participant'], df['Depression_label']))
    
    X, y = [], []
    pt_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.pt')])
    
    print(f"Pooling features from {len(pt_files)} files...")
    for f in pt_files:
        pid = int(f.split('_')[0])
        if pid not in labels_dict: continue
        
        path = os.path.join(features_dir, f)
        features = torch.load(path, map_location='cpu', weights_only=True).float()
        
        # Global Mean Pooling [N, 399, 1024] -> [1024]
        pooled = features.mean(dim=(0, 1)).numpy()
        X.append(pooled)
        y.append(labels_dict[pid])
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset Size: {len(X)} | Features: {X.shape[1]}")
    print(f"Label Distribution: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("SOTA LINEAR PROBE RESULTS")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Balanced Acc: {b_acc*100:.2f}%")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=['No Depression', 'Depression']))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    args = parser.parse_args()
    validate_sota(args.features, args.labels)
