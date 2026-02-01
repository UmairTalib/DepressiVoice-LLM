
import os
import json
from pathlib import Path
from tqdm import tqdm
import re

def identify_residuals():
    base_dir = Path("/home/g755389/depressivoice_phase2_outputs/session_outputs")
    
    # Strict V3 Lexicon
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
    
    residuals = []
    
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 3])
    
    print(f"Scanning {len(sessions)} sessions for residuals...")
    
    for sdir in sessions:
        txt_path = sdir / "cleaned_patient_text_v2.txt" # This is V3 output
        if not txt_path.exists(): continue
        
        with open(txt_path, 'r') as f:
            content = f.read().lower()
            
        hits = []
        for phrase in ellie_lexicon:
            if re.search(rf"\b{re.escape(phrase)}\b", content):
                hits.append(phrase)
        
        if hits:
            residuals.append({
                "session_id": sdir.name,
                "hits": hits
            })
            
    print(f"Found {len(residuals)} residual sessions.")
    print(json.dumps(residuals, indent=2))
    
    with open("/home/g755389/depressivoice_phase2_outputs/residual_sessions.json", "w") as f:
        json.dump(residuals, f, indent=2)

if __name__ == "__main__":
    identify_residuals()
