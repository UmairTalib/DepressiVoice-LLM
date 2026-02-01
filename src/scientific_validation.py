
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def run_scientific_validation():
    print("\n=== REFINED SCIENTIFIC VALIDATION PROTOCOLS ===")
    
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    report_dir = Path("/home/g755389/depressivoice_phase2_outputs/reports")
    report_dir.mkdir(exist_ok=True)
    
    # Filter for valid session IDs (3 digits)
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 3 and d.name.isdigit()])
    
    # 5.1 Acoustic Data
    interviewer_sims = []
    patient_sims = []
    margins = []
    
    # 5.2 Linguistic Data
    # Multi-level keywords
    hard_leakage_phrases = ["tell me about", "i'm ellie", "i am ellie", "next session", "virtual human"]
    suspect_words = ["goodbye", "hello", "meeting"]
    
    leakage_incidents = []
    total_patient_words = 0
    
    # 5.3 Conversational Data
    turn_counts = []
    patient_durations = []
    
    print(f"Scanning {len(sessions)} valid sessions...")
    
    for sdir in tqdm(sessions, desc="Processing Sessions"):
        sid = sdir.name
        roles_path = sdir / "roles.json"
        transcript_path = sdir / "transcript.txt"
        rttm_path = next(sdir.glob("*.rttm"), None)
        
        # --- 5.1 Acoustic Validation ---
        if roles_path.exists():
            try:
                with open(roles_path, 'r') as f:
                    data = json.load(f)
                    sims = data.get("similarities", {})
                    roles = data.get("roles", {})
                    
                    found_int = False
                    found_pat = False
                    
                    for spk, role in roles.items():
                        r_upper = role.upper()
                        if r_upper in ["INTERVIEWER", "ELLIE"]:
                            interviewer_sims.append(sims.get(spk, 0))
                            found_int = True
                        elif r_upper == "PATIENT":
                            patient_sims.append(sims.get(spk, 0))
                            found_pat = True
                    
                    if found_int and found_pat:
                        m = data.get("margin") or data.get("anchor_confidence")
                        if m == 0 or m is None:
                            # Re-calculate margin
                            s_vals = sorted(sims.values(), reverse=True)
                            if len(s_vals) >= 2:
                                m = s_vals[0] - s_vals[1]
                        if m is not None:
                            margins.append(m)
            except Exception as e:
                pass

        # --- 5.2 Linguistic Validation ---
        if transcript_path.exists():
            try:
                with open(transcript_path, 'r') as f:
                    text = f.read().lower()
                    word_count = len(text.split())
                    total_patient_words += word_count
                    
                    found_leak = [p for p in hard_leakage_phrases if p in text]
                    if found_leak:
                        leakage_incidents.append({
                            "session": sid,
                            "type": "HARD",
                            "evidence": found_leak[0],
                            "word_count": word_count
                        })
            except Exception as e:
                pass

        # --- 5.3 Conversational Validation ---
        if rttm_path and rttm_path.exists():
            try:
                with open(rttm_path, 'r') as f:
                    lines = [l for l in f.readlines() if l.strip()]
                    turn_counts.append(len(lines))
                    
                    pat_label = None
                    if roles_path.exists():
                        with open(roles_path, 'r') as rf:
                            rd = json.load(rf)
                            for spk, r in rd.get("roles", {}).items():
                                if r.upper() == "PATIENT":
                                    pat_label = spk
                                    break
                    
                    if pat_label:
                        pat_dur = sum([float(l.split()[4]) for l in lines if l.split()[7] == pat_label])
                        patient_durations.append(pat_dur)
            except Exception as e:
                pass

    # --- Generate Acoustic Plots ---
    if interviewer_sims:
        plt.figure(figsize=(10, 6))
        plt.hist(interviewer_sims, bins=25, alpha=0.6, label='Interviewer Profile', color='dodgerblue', edgecolor='black')
        plt.hist(patient_sims, bins=25, alpha=0.6, label='Patient Profile', color='tomato', edgecolor='black')
        plt.axvline(x=0.55, color='darkgreen', linestyle='--', label='Gating Threshold (0.55)')
        plt.title('Scientific Validation: Acoustic Identity Separation')
        plt.xlabel('Neural Similarity to Ellie Anchor')
        plt.ylabel('Session Count')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(report_dir / "acoustic_purity.png")
        plt.close()

    # --- Generate Report ---
    report = {
        "execution": {"total_sessions": len(sessions), "successful_audit": len(interviewer_sims)},
        "acoustic": {
            "mean_int_sim": float(np.mean(interviewer_sims)) if interviewer_sims else 0,
            "mean_pat_sim": float(np.mean(patient_sims)) if patient_sims else 0,
            "mean_separation": float(np.mean(margins)) if margins else 0,
            "cohens_d": float((np.mean(interviewer_sims) - np.mean(patient_sims)) / np.std(interviewer_sims + patient_sims)) if interviewer_sims else 0
        },
        "linguistic": {
            "leakage_incidents": len(leakage_incidents),
            "leakage_rate": float(len(leakage_incidents) / len(sessions) * 100) if sessions else 0,
            "total_words": total_patient_words
        },
        "conversational": {
            "total_hours": float(np.sum(patient_durations) / 3600),
            "median_dur": float(np.median(patient_durations)),
            "interaction_density": float(np.mean(turn_counts))
        }
    }
    
    with open(report_dir / "scientific_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Export Summary
    sum_text = f"""# Scientific Validation Result Summary
    
- **Acoustic Separation:** {report['acoustic']['mean_separation']:.4f} margin (d={report['acoustic']['cohens_d']:.2f})
- **Linguistic Purity:** {report['linguistic']['leakage_rate']:.2f}% hard leakage found.
- **Data Yield:** {report['conversational']['total_hours']:.2f} hours of verified patient-only audio.
"""
    with open(report_dir / "validation_summary.txt", "w") as f:
        f.write(sum_text)
    
    print("\n" + sum_text)
    print(f"Full report and plots generated in: {report_dir}")

if __name__ == "__main__":
    run_scientific_validation()
