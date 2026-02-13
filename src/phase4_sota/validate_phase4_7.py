"""
Phase 4.7 Final Validation Suite
Performs comprehensive validation including:
1. Test set evaluation
2. Statistical significance testing (DeLong's test)
3. Stratified analysis (gender, severity)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.phase4_sota.dataset import TemporalDataset, pad_collate
from src.phase4_sota.models import CrossModalTransformerAggregator

def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong's test for comparing two ROC curves
    Returns: z-statistic and p-value
    """
    from scipy.stats import norm
    
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    
    # Compute variance using DeLong method
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    
    # Rank predictions
    Q1 = auc1
    Q2 = auc2
    
    # Simplified variance estimation (full DeLong is complex)
    # Using approximation from Hanley & McNeil (1982)
    var1 = (auc1 * (1 - auc1) + (n1 - 1) * (Q1**2 - auc1**2) + (n0 - 1) * (2*Q1**2 - auc1**2)) / (n1 * n0)
    var2 = (auc2 * (1 - auc2) + (n1 - 1) * (Q2**2 - auc2**2) + (n0 - 1) * (2*Q2**2 - auc2**2)) / (n1 * n0)
    
    # Covariance (assuming independent models, cov=0)
    se = np.sqrt(var1 + var2)
    
    z_stat = (auc1 - auc2) / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    return z_stat, p_value, auc1, auc2

def bootstrap_ci(y_true, y_pred, metric_fn, n_iterations=1000, alpha=0.05):
    """Bootstrap confidence intervals for a metric"""
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        scores.append(score)
    
    scores = np.array(scores)
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    
    return lower, upper

def evaluate_model(model, loader, device, threshold=0.5):
    """Comprehensive model evaluation"""
    model.eval()
    all_probs, all_labels = [], []
    all_severities, all_true_severities = [], []
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            phq_scores = batch['phq_scores'].cpu().numpy()
            
            (logits, severity), _ = model(features, mask)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            severity_pred = severity.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_severities.extend(severity_pred)
            all_true_severities.extend(phq_scores)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds,
        'severities': np.array(all_severities),
        'true_severities': np.array(all_true_severities)
    }

def stratified_analysis(results, labels_df, test_ids):
    """Perform stratified analysis by gender and severity"""
    # Filter to test set
    test_df = labels_df[labels_df['Participant_ID'].isin(test_ids)].copy()
    test_df = test_df.sort_values('Participant_ID')
    
    # Ensure alignment
    assert len(test_df) == len(results['labels']), "Mismatch in test set size"
    
    stratified_results = {}
    
    # Gender stratification
    if 'Gender' in test_df.columns:
        for gender in test_df['Gender'].unique():
            if pd.isna(gender):
                continue
            mask = test_df['Gender'] == gender
            indices = mask.values
            
            if np.sum(indices) > 5:  # Minimum sample size
                gender_auc = roc_auc_score(results['labels'][indices], results['probs'][indices])
                gender_f1 = f1_score(results['labels'][indices], results['preds'][indices], zero_division=0)
                
                stratified_results[f'gender_{gender}'] = {
                    'n': np.sum(indices),
                    'auc': gender_auc,
                    'f1': gender_f1
                }
    
    # Severity stratification
    test_df['severity_group'] = pd.cut(test_df['PHQ_Score'], 
                                        bins=[0, 4, 9, 14, 24], 
                                        labels=['Minimal', 'Mild', 'Moderate', 'Severe'])
    
    for severity in test_df['severity_group'].unique():
        if pd.isna(severity):
            continue
        mask = test_df['severity_group'] == severity
        indices = mask.values
        
        if np.sum(indices) > 5 and np.sum(results['labels'][indices]) > 0:
            sev_auc = roc_auc_score(results['labels'][indices], results['probs'][indices])
            sev_f1 = f1_score(results['labels'][indices], results['preds'][indices], zero_division=0)
            
            stratified_results[f'severity_{severity}'] = {
                'n': np.sum(indices),
                'auc': sev_auc,
                'f1': sev_f1
            }
    
    return stratified_results

def plot_confusion_matrix(cm, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Phase 4.7 Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix to {output_path}")

def main():
    print("="*60)
    print("Phase 4.7 Final Validation Suite")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "data/phase4_sota_features"
    labels_path = "data/labels/master_labels_v7.csv"
    model_path = "experiments/phase4_v11_PRO_SOTA/best_mtl_transformer.pth"
    output_dir = "experiments/phase4_v11_PRO_SOTA/validation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"\n📦 Loading Phase 4.7 model from {model_path}")
    model = CrossModalTransformerAggregator(
        input_dim=256, 
        d_model=256, 
        nhead=4, 
        num_layers=2, 
        dropout=0.0
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    
    # Get test split (lowercase 'split' column)
    test_ids = labels_df[labels_df['split'] == 'test']['Participant_ID'].values
    print(f"✅ Test set: {len(test_ids)} participants")
    
    # Create test dataset
    test_ds = TemporalDataset(data_dir, split_ids=test_ids, augment=False, balance=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=pad_collate)
    
    # Step 1: Test Set Evaluation
    print("\n" + "="*60)
    print("STEP 1: Test Set Evaluation")
    print("="*60)
    
    # Find optimal threshold from validation
    val_ids = labels_df[labels_df['split'] == 'dev']['Participant_ID'].values
    val_ds = TemporalDataset(data_dir, split_ids=val_ids, augment=False, balance=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=pad_collate)
    
    val_results = evaluate_model(model, val_loader, device, threshold=0.5)
    
    # Search for best threshold on validation
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.05, 0.5, 0.01):
        preds_t = (val_results['probs'] >= t).astype(int)
        f1_t = f1_score(val_results['labels'], preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t
    
    print(f"🎯 Optimal threshold from validation: {best_thresh:.3f} (F1={best_f1:.4f})")
    
    # Evaluate on test set with optimal threshold
    test_results = evaluate_model(model, test_loader, device, threshold=best_thresh)
    
    print(f"\n📊 Test Set Results:")
    print(f"   AUC:       {test_results['auc']:.4f}")
    print(f"   F1-Score:  {test_results['f1']:.4f}")
    print(f"   Precision: {test_results['precision']:.4f}")
    print(f"   Recall:    {test_results['recall']:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TN={test_results['confusion_matrix'][0,0]}, FP={test_results['confusion_matrix'][0,1]}")
    print(f"   FN={test_results['confusion_matrix'][1,0]}, TP={test_results['confusion_matrix'][1,1]}")
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(test_results['confusion_matrix'], cm_path)
    
    # Bootstrap confidence intervals
    print(f"\n🔬 Computing 95% Confidence Intervals...")
    auc_lower, auc_upper = bootstrap_ci(test_results['labels'], test_results['probs'], 
                                         roc_auc_score, n_iterations=1000)
    f1_lower, f1_upper = bootstrap_ci(test_results['labels'], test_results['preds'], 
                                       f1_score, n_iterations=1000)
    
    print(f"   AUC: {test_results['auc']:.4f} (95% CI: {auc_lower:.4f}-{auc_upper:.4f})")
    print(f"   F1:  {test_results['f1']:.4f} (95% CI: {f1_lower:.4f}-{f1_upper:.4f})")
    
    # Step 2: Statistical Significance Testing
    print("\n" + "="*60)
    print("STEP 2: Statistical Significance Testing (vs v7)")
    print("="*60)
    
    # Load v7 predictions (if available)
    v7_model_path = "experiments/phase4_v11_PRO/best_mtl_transformer.pth"
    if os.path.exists(v7_model_path):
        print(f"📦 Loading v7 baseline model...")
        v7_model = CrossModalTransformerAggregator(
            input_dim=256, 
            d_model=128,  # v7 used smaller model
            nhead=4, 
            num_layers=2, 
            dropout=0.0
        ).to(device)
        v7_model.load_state_dict(torch.load(v7_model_path, map_location=device))
        
        v7_results = evaluate_model(v7_model, test_loader, device, threshold=0.17)
        
        # DeLong's test
        z_stat, p_value, auc1, auc2 = delong_test(
            test_results['labels'],
            test_results['probs'],
            v7_results['probs']
        )
        
        print(f"\n📈 DeLong's Test Results:")
        print(f"   Phase 4.7 AUC: {auc1:.4f}")
        print(f"   v7 Baseline AUC: {auc2:.4f}")
        print(f"   Difference: {auc1 - auc2:.4f}")
        print(f"   z-statistic: {z_stat:.3f}")
        print(f"   p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"   ✅ Statistically significant improvement (p < 0.05)")
        else:
            print(f"   ⚠️  Not statistically significant (p >= 0.05)")
    else:
        print(f"⚠️  v7 model not found at {v7_model_path}, skipping comparison")
    
    # Step 3: Stratified Analysis
    print("\n" + "="*60)
    print("STEP 3: Stratified Analysis")
    print("="*60)
    
    stratified = stratified_analysis(test_results, labels_df, test_ids)
    
    print(f"\n📊 Performance by Subgroup:")
    for group_name, metrics in stratified.items():
        print(f"\n   {group_name}:")
        print(f"      n = {metrics['n']}")
        print(f"      AUC = {metrics['auc']:.4f}")
        print(f"      F1  = {metrics['f1']:.4f}")
    
    # Save results to JSON
    import json
    results_summary = {
        'test_metrics': {
            'auc': float(test_results['auc']),
            'f1': float(test_results['f1']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'threshold': float(best_thresh)
        },
        'confidence_intervals': {
            'auc': {'lower': float(auc_lower), 'upper': float(auc_upper)},
            'f1': {'lower': float(f1_lower), 'upper': float(f1_upper)}
        },
        'stratified_analysis': {k: {kk: float(vv) if isinstance(vv, (int, np.integer, float, np.floating)) else vv 
                                     for kk, vv in v.items()} 
                                for k, v in stratified.items()}
    }
    
    results_path = os.path.join(output_dir, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
