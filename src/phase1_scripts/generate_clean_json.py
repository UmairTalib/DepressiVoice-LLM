
import os
import json
import re
from pathlib import Path
from tqdm import tqdm

def generate_clean_json():
    print("\n=== STARTING JSON METADATA REPAIR (V3 FILTER) ===")
    
    # Target Local Processed Data
    base_dir = Path("data/processed/phase1/session_outputs")
    
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return

    # STRICT Lexicon (V3 Protocol)
    # Must match clean_transcripts_v3.py EXACTLY
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
    
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 3])
    count = 0
    
    for sdir in tqdm(sessions, desc="Reparing JSONs"):
        # Source: The ORIGINAL transcript (which contains timestamps)
        # We assume 'transcript.json' is the Whisper output
        # If not found, try to find *any* transcript json
        json_path = sdir / "transcript.json"
        
        if not json_path.exists():
            # Fallback for weird naming
            candidates = list(sdir.glob("*transcript*.json"))
            # avoid cleaned_transcript
            candidates = [c for c in candidates if "cleaned" not in c.name]
            if candidates:
                json_path = candidates[0]
            else:
                continue
                
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                segments = data.get("segments", [])
                
            clean_segments = []
            
            for seg in segments:
                text = seg.get("text", "").strip()
                text_lower = text.lower()
                
                # Check 1: Confidence (V3 Logic)
                logprob = seg.get("avg_logprob", 0)
                if logprob < -0.7 and len(text.split()) < 3:
                    continue # Drop noise

                # Check 2: Aggressive Lexicon Match
                is_leakage = False
                for phrase in ellie_lexicon:
                    # Regex boundary match
                    if re.search(rf"\b{re.escape(phrase)}\b", text_lower):
                        is_leakage = True
                        break
                
                if not is_leakage:
                    clean_segments.append(seg)
            
            # Save CLEAN JSON
            # This is the critical missing file for Phase 3 loader
            output_json = sdir / "cleaned_transcript.json"
            
            with open(output_json, 'w') as f:
                json.dump({"segments": clean_segments}, f, indent=2)
                
            count += 1
            
        except Exception as e:
            print(f"Error {sdir.name}: {e}")

    print(f"Repair Complete. Generated {count} 'cleaned_transcript.json' files.")

if __name__ == "__main__":
    generate_clean_json()
