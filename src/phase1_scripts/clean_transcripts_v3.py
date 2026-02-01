
import os
import json
import csv
from pathlib import Path
from tqdm import tqdm
import re

def run_cleaning_v3():
    print("\n=== STARTING TRANSCRIPT CLEANING V3 (AGGRESSIVE) ===")
    
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    report_dir = Path("/home/g755389/depressivoice_phase2_outputs/cleanup_reports")
    report_dir.mkdir(exist_ok=True)
    
    # STRICT Lexicon (High Confidence Leakage)
    # Removing ambiguous one-worders.
    ellie_lexicon = [
        "how are you doing today", "how are you doing", "how are you", 
        "where are you from originally", "tell me about yourself",
        "tell me about", "what do you mean", "can you explain",
        "why do you think", "thank you for sharing", "thanks for sharing",
        "when was the last time", "how close are you to",
        "what are you like", "what do you enjoy", "what do you do",
        "have you been diagnosed", "have you noticed",
        "what advice would you give", "how easy is it to",
        "how hard is it to", "what are some things",
        "hi i'm ellie", "i am ellie", "i'm a virtual interviewer",
        "created to talk to people", "i'm not a therapist",
        "your answers are anonymous",
        "feel free to say", "whenever you're ready",
        "thanks for talking", "have a good day"
    ]
    
    total_cleaned = 0
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 3])

    for sdir in tqdm(sessions, desc="Cleaning V3"):
        json_path = sdir / "transcript.json"
        
        if not json_path.exists(): continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                segments = data.get("segments", [])
                
            clean_segments = []
            
            for seg in segments:
                text = seg.get("text", "").strip()
                text_lower = text.lower()
                
                # Check 1: Confidence
                logprob = seg.get("avg_logprob", 0)
                if logprob < -0.7 and len(text.split()) < 3:
                    continue # Drop noise

                # Check 2: Aggressive Lexicon Match
                is_leakage = False
                for phrase in ellie_lexicon:
                    # Regex boundary match to avoid substring errors
                    # e.g. avoid matching "shell" for "ell" (not applicable here but good practice)
                    if re.search(rf"\b{re.escape(phrase)}\b", text_lower):
                        is_leakage = True
                        break
                
                if not is_leakage:
                    clean_segments.append(seg)
            
            # Save V3 Output (Overwriting V2 for simplicity in final workflow)
            data["segments"] = clean_segments
            full_text_v3 = " ".join([s["text"].strip() for s in clean_segments])
            
            with open(sdir / "cleaned_patient_text_v2.txt", "w") as f:
                 f.write(full_text_v3)
                 
            total_cleaned += 1
            
        except Exception as e:
            print(f"Error {sdir.name}: {e}")

    print(f"V3 Aggressive Cleaning Complete. Processed {total_cleaned} sessions.")

if __name__ == "__main__":
    run_cleaning_v3()
