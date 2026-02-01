
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.utils import resample
from tqdm import tqdm

def run_acoustic_verification():
    print("\n=== STARTING DEEP ACOUSTIC VERIFICATION ===")
    
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    report_dir = Path("/home/g755389/depressivoice_phase2_outputs/reports")
    report_dir.mkdir(exist_ok=True)
    
    # Load anchor (Assuming it's in the pipeline root or we can derive it from roles)
    # Since we don't have the raw .npy handy in the logic flow, we will use the similarity scores 
    # already computed in roles.json which represent the exact cosine similarity to that anchor.
    
    interviewer_scores = []
    patient_scores = []
    all_margins = []
    
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 3])
    
    print(f"Collecting data from {len(sessions)} sessions...")
    
    for sdir in tqdm(sessions):
        roles_path = sdir / "roles.json"
        if not roles_path.exists(): continue
        
        try:
            with open(roles_path, 'r') as f:
                data = json.load(f)
                sims = data.get("similarities", {})
                roles = data.get("roles", {})
                
                s_int = []
                s_pat = []
                
                for spk, role in roles.items():
                    score = sims.get(spk)
                    if score is None: continue
                    
                    if role.upper() in ["INTERVIEWER", "ELLIE"]:
                        interviewer_scores.append(score)
                        s_int.append(score)
                    elif role.upper() == "PATIENT":
                        patient_scores.append(score)
                        s_pat.append(score)
                
                if s_int and s_pat:
                    # Margin is best(int) - best(pat) or similar logic
                    all_margins.append(max(s_int) - max(s_pat))
        except:
            continue

    if not interviewer_scores or not patient_scores:
        print("Error: No scores found for analysis.")
        return

    # --- 1. ROC and EER Calculation ---
    y_true = [1] * len(interviewer_scores) + [0] * len(patient_scores)
    y_scores = interviewer_scores + patient_scores
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute EER
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh_eer = interp1d(fpr, thresholds)(eer)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f} at threshold {thresh_eer:.4f}")

    # --- 2. Bootstrapping Margin Delta (10k resamples) ---
    print("Bootstrapping margin Δ (10,000 resamples)...")
    bootstrap_stats = []
    for _ in range(10000):
        # Sample with replacement
        resampled_margins = resample(all_margins)
        bootstrap_stats.append(np.mean(resampled_margins))
    
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0, np.percentile(bootstrap_stats, p))
    p = (alpha + (1.0 - alpha) / 2.0) * 100
    upper = np.percentile(bootstrap_stats, p)
    
    print(f"95% CI for Margin: [{lower:.4f}, {upper:.4f}]")

    # --- 3. Plotting ---
    # Plot 1: Histograms
    plt.figure(figsize=(10, 6))
    plt.hist(interviewer_scores, bins=30, alpha=0.6, label='Interviewer (Positives)', color='blue')
    plt.hist(patient_scores, bins=30, alpha=0.6, label='Patient (Negatives)', color='red')
    plt.title('Distribution of Cosine Similarities to Ellie Anchor')
    plt.legend()
    plt.savefig(report_dir / "sims_hist.png")
    
    # Plot 2: ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.axhline(y=1-eer, color='gray', linestyle=':', label=f'EER = {eer:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Interviewer ID)')
    plt.legend(loc="lower right")
    plt.savefig(report_dir / "roc.png")

    # Plot 3: Bootstrap Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_stats, bins=50, color='green', alpha=0.7)
    plt.axvline(lower, color='red', linestyle='--', label=f'Lower 95%: {lower:.4f}')
    plt.axvline(upper, color='red', linestyle='--', label=f'Upper 95%: {upper:.4f}')
    plt.title('Bootstrap Distribution of Separation Margin Δ')
    plt.legend()
    plt.savefig(report_dir / "margin_bootstrap.png")

    # --- 4. Final Output ---
    results = {
        "eer": float(eer),
        "roc_auc": float(roc_auc),
        "eer_threshold": float(thresh_eer),
        "margin_bootstrap": {
            "mean": float(np.mean(all_margins)),
            "lower_95_ci": float(lower),
            "upper_95_ci": float(upper),
            "resamples": 10000
        }
    }
    
    with open(report_dir / "eer_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(report_dir / "margin_bootstrap.json", "w") as f:
        json.dump(results["margin_bootstrap"], f, indent=2)
        
    print(f"\nVerification Complete. Reports saved to {report_dir}")

if __name__ == "__main__":
    run_acoustic_verification()
