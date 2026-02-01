
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import re

def run_purity_validations():
    print("\n=== STARTING LINGUISTIC & CONVERSATIONAL VALIDATIONS ===")
    
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    output_dir = Path("/home/g755389/depressivoice_phase2_outputs/validation_final_bundle")
    output_dir.mkdir(exist_ok=True)
    
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 3])
    print(f"Processing {len(sessions)} sessions...")

    # --- PART 1: LINGUISTIC PURITY ---
    print("\n[Step 1] Linguistic Purity Analysis...")
    ellie_prompts = [
        "how are you feeling", "tell me about", "what do you mean", 
        "can you explain", "why do you think", "thank you for sharing",
        "hi i'm ellie", "i am ellie", "glad to meet you", "how are you today",
        "i was built to", "i'm interested in"
    ]
    
    contamination_records = []
    session_has_contamination = 0
    total_scanned = 0
    keyword_hits = {}

    for sdir in tqdm(sessions, desc="Linguistic"):
        txt_path = sdir / "transcript.txt"
        if not txt_path.exists(): continue
        
        total_scanned += 1
        with open(txt_path, 'r') as f:
            content = f.read().lower()
            
            sid_hits = []
            for prompt in ellie_prompts:
                matches = re.findall(rf"\b{re.escape(prompt)}\b", content)
                if matches:
                    count = len(matches)
                    sid_hits.append({"phrase": prompt, "count": count})
                    keyword_hits[prompt] = keyword_hits.get(prompt, 0) + count
            
            if sid_hits:
                session_has_contamination += 1
                contamination_records.append({"session": sdir.name, "hits": sid_hits})

    # Save Linguistic Artifacts
    purity_stats = {
        "total_sessions_scanned": total_scanned,
        "contaminated_sessions": session_has_contamination,
        "contamination_rate_percent": (session_has_contamination / total_scanned * 100) if total_scanned > 0 else 0,
        "top_matched_phrases": sorted(keyword_hits.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    
    with open(output_dir / "linguistic_purity_report.json", "w") as f:
        json.dump(purity_stats, f, indent=2)
        
    pd.DataFrame(contamination_records).to_csv(output_dir / "linguistic_keyword_hits.csv", index=False)
    
    # Histogram of hits per phrase
    if keyword_hits:
        plt.figure(figsize=(10, 6))
        phrases = [x[0] for x in purity_stats["top_matched_phrases"]]
        counts = [x[1] for x in purity_stats["top_matched_phrases"]]
        plt.barh(phrases, counts, color='teal')
        plt.xlabel("Total Hits Across Dataset")
        plt.title("Top Linguistic Contamination Matches")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / "linguistic_histogram.png")

    # --- PART 2: CONVERSATIONAL STRUCTURE ---
    print("\n[Step 2] Conversational Metrics Analysis...")
    conv_data = []

    for sdir in tqdm(sessions, desc="Conversational"):
        rttm_path = next(sdir.glob("*.rttm"), None)
        if not rttm_path: continue
        
        turns = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 8: continue
                # Start, Duration, Speaker
                turns.append({"start": float(parts[3]), "dur": float(parts[4]), "spk": parts[7]})
        
        if not turns: continue
        
        df_turns = pd.DataFrame(turns)
        total_time = df_turns['start'].max() + df_turns['dur'].iloc[-1]
        turn_count = len(df_turns)
        avg_dur = df_turns['dur'].mean()
        density = (turn_count / total_time) * 60 if total_time > 0 else 0
        
        conv_data.append({
            "session": sdir.name,
            "turn_count": turn_count,
            "avg_turn_duration": avg_dur,
            "turn_density": density,
            "total_duration": total_time
        })

    df_conv = pd.DataFrame(conv_data)
    
    # Outlier Detection (Z-score > 3)
    density_mean = df_conv['turn_density'].mean()
    density_std = df_conv['turn_density'].std()
    df_conv['density_z'] = (df_conv['turn_density'] - density_mean) / density_std
    outliers = df_conv[df_conv['density_z'].abs() > 3]
    
    # Save Conversational Artifacts
    conv_report = {
        "avg_turns_per_session": float(df_conv['turn_count'].mean()),
        "avg_turn_duration": float(df_conv['avg_turn_duration'].mean()),
        "avg_turn_density_tpm": float(density_mean),
        "outlier_count": len(outliers)
    }
    
    with open(output_dir / "conversational_metrics.json", "w") as f:
        json.dump(conv_report, f, indent=2)
        
    outliers.to_csv(output_dir / "session_outliers.csv", index=False)
    
    # Turn Density Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df_conv['turn_density'], bins=30, color='royalblue', edgecolor='black', alpha=0.7)
    plt.axvline(density_mean, color='red', linestyle='--', label=f'Mean: {density_mean:.2f}')
    plt.xlabel("Turns Per Minute (TPM)")
    plt.ylabel("Session Count")
    plt.title("Conversational Density Distribution")
    plt.legend()
    plt.savefig(output_dir / "turn_density_histogram.png")

    print(f"\nValidations Complete. Artifacts saved to: {output_dir}")

if __name__ == "__main__":
    run_purity_validations()
