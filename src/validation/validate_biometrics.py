
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def validate_biometrics(data_root):
    """
    Analyzes biometric similarity scores to validate the robustness of speaker role assignment.
    """
    print("INFO: Starting Biometric Validation...")
    
    # Path to sidecar JSONs
    json_pattern = os.path.join(data_root, "*_biometrics.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"WARNING: No biometric validation files found in {data_root}.")
        return

    records = []
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            print(f"WARNING: Error reading {jf}: {e}")

    df = pd.DataFrame(records)
    
    # --- METRIC 1: SIMILARITY GAP (DELTA) ---
    margins = df['margin'].values
    mean_margin = np.mean(margins)
    min_margin = np.min(margins)
    risky_sessions = df[df['margin'] < 0.05]
    
    # --- METRIC 2: CLUSTER COMPACTNESS ---
    # Collect all Interviewer Scores vs Patient Scores
    # Note: 'scores' is a dict {spk_id: score}
    # 'interviewer_identified' is the spk_id of the interviewer
    
    interviewer_scores = []
    patient_scores = []
    
    for _, row in df.iterrows():
        int_id = row['interviewer_identified']
        scores = row['scores']
        for spk, score in scores.items():
            if spk == int_id:
                interviewer_scores.append(score)
            else:
                patient_scores.append(score)

    int_mean = np.mean(interviewer_scores)
    int_std = np.std(interviewer_scores)
    pat_mean = np.mean(patient_scores)
    pat_std = np.std(patient_scores)

    # --- REPORT GENERATION ---
    report = f"""# Biometric Validation Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sessions Analyzed:** {len(df)}

## 1. Reliability (Similarity Gap)
The "Similarity Gap" ($\Delta$) is the difference between the Interviewer's match score and the Patient's match score.
*   **Mean Gap:** {mean_margin:.4f} (Target: > 0.1)
*   **Min Gap:** {min_margin:.4f}
*   **Risky Sessions ($\Delta < 0.05$):** {len(risky_sessions)} ({len(risky_sessions)/len(df)*100:.1f}%)

## 2. Cluster Analysis
*   **Interviewer Cluster:** Mean={int_mean:.4f}, Std={int_std:.4f}
    *   *Interpretation:* Should be High Mean, Low Std (Tight cluster around Anchor).
*   **Patient Cluster:** Mean={pat_mean:.4f}, Std={pat_std:.4f}
    *   *Interpretation:* Should be Low Mean, High Std (Wide distribution).

## 3. Conclusion
{'✅ PASSED: System is discriminating identities effectively.' if mean_margin > 0.1 else '⚠️ WARNING: Margins are narrow. Check anchor quality.'}
"""

    report_path = "docs/BIOMETRIC_VALIDATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
        
    print(report)
    print(f"INFO: Report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/02_transcribed", help="Path to JSONs")
    args = parser.parse_args()
    
    validate_biometrics(args.data_root)
