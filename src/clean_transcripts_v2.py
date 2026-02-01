
import os
import json
import math
import csv
from pathlib import Path
from tqdm import tqdm
import re

def run_cleaning_v2():
    print("\n=== STARTING TRANSCRIPT CLEANING V2 ===")
    
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    report_dir = Path("/home/g755389/depressivoice_phase2_outputs/cleanup_reports")
    report_dir.mkdir(exist_ok=True)
    
    # Ellie's known static prompts (The "Interviewer Set")
    # Derived from DAIC-WOZ protocols
    ellie_lexicon = [
        "how are you doing today", "how are you doing", "how are you", 
        "where are you from originally", "tell me about yourself",
        "tell me about", "what do you mean", "can you explain",
        "why do you think", "thank you for sharing", "thanks for sharing",
        "i see", "uh huh", "go on", "okay", "cool", "awesome",
        "when was the last time", "how close are you to",
        "what are you like", "what do you enjoy", "what do you do",
        "have you been diagnosed", "have you noticed",
        "what advice would you give", "how easy is it to",
        "how hard is it to", "what are some things",
        "hi i'm ellie", "i am ellie", "i'm a virtual interviewer",
        "created to talk to people", "i'm not a therapist",
        "don't worry", "your answers are anonymous",
        "feel free to say", "whenever you're ready",
        "goodbye", "thanks for talking", "have a good day"
    ]
    
    # Compile regex for efficiency
    # Match phrases that appear as full sentences or distinct clauses
    # Using simple inclusion for now, but checking boundaries
    
    manifest_rows = []
    total_cleaned_sessions = 0
    
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 3])
    
    print(f"Processing {len(sessions)} sessions...")
    
    for sdir in tqdm(sessions):
        sid = sdir.name
        json_path = sdir / "transcript.json"
        
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                segments = data.get("segments", [])
                
            clean_segments = []
            removed_segments_count = 0
            reason_counts = {"lexicon": 0, "confidence": 0}
            
            for seg in segments:
                text = seg.get("text", "").strip()
                text_lower = text.lower()
                
                # Check 1: Confidence (avg_logprob < -0.7 => < 50%)
                # Only apply if text is short (likely noise/muttering)
                logprob = seg.get("avg_logprob", 0)
                is_low_conf = logprob < -0.7
                
                # Check 2: Lexicon Match
                is_leakage = False
                matched_phrase = ""
                
                # Exact/Fuzzy match
                for phrase in ellie_lexicon:
                    # If the segment contains the phrase significantly
                    if phrase in text_lower:
                        # Heuristic: If phrase is > 50% of the segment length, drop it
                        if len(phrase) / len(text_lower) > 0.6:
                            is_leakage = True
                            matched_phrase = phrase
                            break
                        # Or if it's an exact match
                        if text_lower == phrase:
                            is_leakage = True
                            matched_phrase = phrase
                            break
                
                if is_leakage:
                    removed_segments_count += 1
                    reason_counts["lexicon"] += 1
                elif is_low_conf and len(text.split()) < 3:
                     # Remove short low-conf segments (muttering/noise)
                    removed_segments_count += 1
                    reason_counts["confidence"] += 1
                else:
                    clean_segments.append(seg)
            
            # --- Save Outputs ---
            # 1. New V2 JSON
            data["segments"] = clean_segments
            # Rebuild full text
            full_text_v2 = " ".join([s["text"].strip() for s in clean_segments])
            data["text"] = full_text_v2
            
            with open(sdir / "cleaned_patient_text_v2.json", "w") as f:
                json.dump(data, f, indent=2)
                
            # 2. New V2 TXT
            with open(sdir / "cleaned_patient_text_v2.txt", "w") as f:
                 f.write(full_text_v2)
                 
            # 3. Cleanup Report
            report = {
                "session_id": sid,
                "original_segments": len(segments),
                "removed_segments": removed_segments_count,
                "final_segments": len(clean_segments),
                "reasons": reason_counts
            }
            with open(sdir / "cleanup_report.json", "w") as f:
                json.dump(report, f, indent=2)
                
            manifest_rows.append(report)
            total_cleaned_sessions += 1
            
        except Exception as e:
            print(f"Error processing {sid}: {e}")

    # --- Global Manifest ---
    with open(report_dir / "cleanup_manifest.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "original_segments", "removed_segments", "final_segments", "reasons"])
        writer.writeheader()
        for row in manifest_rows:
            # Flatten reasons for CSV
            row_csv = row.copy()
            row_csv["reasons"] = json.dumps(row["reasons"])
            writer.writerow(row_csv)
            
    print(f"\nCleaning Complete. Processed {total_cleaned_sessions} sessions.")
    print(f"Manifest saved to {report_dir / 'cleanup_manifest.csv'}")

if __name__ == "__main__":
    run_cleaning_v2()
