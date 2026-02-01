
import os
import json
import csv
from pathlib import Path
from tqdm import tqdm
import re

def final_audit():
    print("\n=== STARTING FINAL LOCKED AUDIT ===")
    
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    report_dir = Path("/home/g755389/depressivoice_phase2_outputs/phase2_worktree/final_lock_bundle")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Same Strict Lexicon
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
    contaminated_count = 0
    loss_rows = []
    
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 3])

    for sdir in tqdm(sessions):
        # Prefer V4, fallback to V2 (which is V3 result), fallback to transcript
        txt_path = sdir / "cleaned_patient_text_v4.txt"
        if not txt_path.exists():
            txt_path = sdir / "cleaned_patient_text_v2.txt"
        
        if not txt_path.exists(): continue
            
        with open(txt_path, 'r') as f:
            content = f.read().lower()
            
        # 1. Linguistic Check
        hits = []
        for phrase in ellie_lexicon:
            if re.search(rf"\b{re.escape(phrase)}\b", content):
                hits.append(phrase)
        
        if hits:
            contaminated_count += 1
            
        # 2. Content Loss
        # Get Original (Raw) Word Count
        orig_path = sdir / "transcript.txt"
        if orig_path.exists():
            with open(orig_path, 'r') as f:
                raw_words = len(f.read().split())
        else:
            raw_words = 1
            
        final_words = len(content.split())
        perc_loss = (1 - final_words/raw_words)*100
        
        loss_rows.append({
            "session_id": sdir.name,
            "original_words": raw_words,
            "final_words": final_words,
            "percent_loss": perc_loss,
            "is_flagged": perc_loss > 5.0
        })
        
        total_cleaned += 1

    # Save Reports
    contamination_rate = (contaminated_count / total_cleaned) * 100
    
    audit_summary = {
        "total_sessions": total_cleaned,
        "contaminated_sessions": contaminated_count,
        "rate": contamination_rate,
        "status": "PASS" if contaminated_count <= 1 else "FAIL"
    }
    
    with open(report_dir / "final_audit_summary.json", "w") as f:
        json.dump(audit_summary, f, indent=2)
        
    with open(report_dir / "content_loss_table.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "original_words", "final_words", "percent_loss", "is_flagged"])
        writer.writeheader()
        writer.writerows(loss_rows)
        
    print(f"Final Count: {contaminated_count} contaminated sessions.")
    print(f"Status: {audit_summary['status']}")

if __name__ == "__main__":
    final_audit()
