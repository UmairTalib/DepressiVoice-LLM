
import sys, os, torch
import soundfile as sf
import numpy as np
from pathlib import Path
import json

# Add the pipeline directory to path
sys.path.append('/home/g755389/depressivoice_phase2_outputs')
import phase2_pyannote_sota

def run_ultra_rescue():
    print("\n=== ULTRA RESCUE: Session 309 (Forced Speaker Separation) ===")
    
    # 1. Setup Models
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is missing!")
        return
        
    d, e = phase2_pyannote_sota.load_models(hf_token)
    
    # 2. Define Paths
    root = Path('/home/g755389/DepressiVoice-LLM')
    audio_path = root / "data/01_raw/309_AUDIO.wav"
    output_dir = Path('/home/g755389/depressivoice_phase2_outputs/session_outputs/309')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not audio_path.exists():
        print(f"ERROR: Audio file not found at {audio_path}")
        return

    # 3. FORCE 2 SPEAKERS
    print(f"Diarizing {audio_path.name} with num_speakers=2 (Forcing Split)...")
    diarization = d(str(audio_path), num_speakers=2)
    
    # 4. Create Anchor to identity Ellie
    print("Creating anchor for role validation...")
    wavs = sorted(root.glob('data/01_raw/*_AUDIO.wav'))
    anchor = phase2_pyannote_sota.create_ellie_anchor(wavs, d, e)
    
    # 5. Role Assignment by Similarity
    speakers = diarization.labels()
    print(f"Speakers identified: {speakers}")
    
    similarities = {}
    for spk in speakers:
        segments = [s for s, _, l in diarization.itertracks(yield_label=True) if l == spk]
        emb = phase2_pyannote_sota.extract_speaker_embedding(str(audio_path), segments[:10], e)
        if emb is not None:
             sim = torch.nn.functional.cosine_similarity(
                torch.from_numpy(emb).squeeze(), 
                torch.from_numpy(anchor).squeeze(), 
                dim=0
             ).item()
             similarities[spk] = sim
             print(f"  {spk} Similarity to Ellie: {sim:.4f}")

    if not similarities:
        print("FAILURE: No voice detected at all.")
        return

    # Pick Ellie by highest similarity
    ellie = max(similarities, key=similarities.get)
    patient = next((s for s in speakers if s != ellie), None)
    
    print(f"Targeting: {ellie}=ELLIE, {patient}=PATIENT")
    roles = {ellie: "ELLIE"}
    if patient:
        roles[patient] = "PATIENT"
        
    # 6. Extract Audio
    if patient:
        print(f"Extracting Patient Audio ({patient})...")
        res_data = phase2_pyannote_sota.extract_patient_audio(diarization, roles, str(audio_path))
        if res_data is not None:
            audio, sr = res_data
            out_file = output_dir / f"patient_audio_309.wav"
            sf.write(str(out_file), audio, sr)
            print(f"SUCCESS! Saved rescued audio to {out_file}")
            
            # Save updated roles.json
            with open(output_dir / "roles.json", "w") as f:
                json.dump(roles, f, indent=2)
            print(f"Updated roles.json saved.")
        else:
            print("Failed to extract segments for the identified patient.")
    else:
        print("Could not identify a secondary speaker even with forced split.")

if __name__ == "__main__":
    run_ultra_rescue()
