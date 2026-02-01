
import sys
import os
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import traceback

# Add the pipeline directory to path so we can import the original script
sys.path.append('/home/g755389/depressivoice_phase2_outputs')
try:
    import phase2_pyannote_sota
except ImportError:
    # Just in case we run locally, mock for syntax check
    pass

# --- CAPTURE ORIGINAL REFERENCE (Fix for Recursion Crash) ---
_orig_assign_roles = phase2_pyannote_sota.assign_roles

# --- OVERRIDE 1: Handle Stereo (Fix for 603, 192, etc.) ---
def extract_speaker_embedding_mono_safe(audio_file, segments, embedding_model):
    if not segments:
        return None
    
    # Load audio
    waveform, sr = sf.read(audio_file)
    
    # STEREO FIX: Convert to Mono if needed
    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        # Average the channels
        waveform = np.mean(waveform, axis=1)
    
    # Concatenate all segments
    speaker_audio = []
    for seg in segments:
        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        if start_sample < len(waveform) and end_sample <= len(waveform):
             speaker_audio.append(waveform[start_sample:end_sample])
    
    if not speaker_audio: return None
    
    # Concatenate
    speaker_audio = np.concatenate(speaker_audio)
    
    # Convert to Tensor [1, 1, samples]
    waveform_tensor = torch.from_numpy(speaker_audio).float()
    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)
    
    # Embedding
    embedding = embedding_model(waveform_tensor)
    
    # Numpy fix
    if hasattr(embedding, 'detach'):
        return embedding.detach().cpu().numpy()
    return embedding

# --- OVERRIDE 2: Relaxed Roles (Fix for 309) ---
def assign_roles_relaxed_logging(sid, dr, path, em, anchor):
    # Call ORIGINAL function using saved reference (Prevents Recursion)
    roles, similarities, m, c = _orig_assign_roles(sid, dr, path, em, anchor)
    
    # If standard logic worked
    if 'PATIENT' in roles.values():
        return roles, similarities, m, c
        
    print(f"    [INFO] Standard rule failed for {sid}. Checking for Relaxed candidates...")
    
    # Find Ellie
    ellie = next((k for k, v in roles.items() if v == 'ELLIE'), None)
    if not ellie: return roles, similarities, m, c 
    
    # Find longest other speaker
    others = {l: 0.0 for l in dr.labels() if l != ellie}
    for seg, _, l in dr.itertracks(yield_label=True):
        if l in others: others[l] += seg.duration
            
    if others:
        best_spk = max(others.items(), key=lambda x: x[1])[0]
        roles[best_spk] = 'PATIENT'
        # LOGGING TAG for counting later
        print(f"    [RESCUE_RELAXED] Force-assigned {best_spk} as PATIENT ({others[best_spk]:.1f}s)")
        
    return roles, similarities, m, c

# --- EXECUTION ---
def run_rescue():
    # 1. Apply Patches
    phase2_pyannote_sota.extract_speaker_embedding = extract_speaker_embedding_mono_safe
    phase2_pyannote_sota.assign_roles = assign_roles_relaxed_logging
    
    # FIX: Initialize Globals
    PROJECT_ROOT = Path('/home/g755389/DepressiVoice-LLM')
    OUTPUT_DIR = Path('/home/g755389/depressivoice_phase2_outputs')
    
    # 2. Load Models
    print("Loading models...")
    hf_token = os.environ.get('HF_TOKEN')
    d, e = phase2_pyannote_sota.load_models(hf_token)
    
    # 3. Create Anchor
    print("Creating anchor...")
    wavs = sorted(PROJECT_ROOT.glob('data/01_raw/*_AUDIO.wav'))
    anchor = phase2_pyannote_sota.create_ellie_anchor(wavs, d, e)
    
    # 4. List of Bad Sessions (CORRECTED IDs)
    bad_sessions = [
        "309", "603", "609", "620", "622", "623", "625", "629", 
        "632", "633", "634", "635", "637", "640", "649", "653", 
        "657", "659", "661", "662", "666", "669", "677", "680", 
        "683", "684", "689", "691", "692", "695", "697", "698", 
        "707", "709", "717"
    ]
    
    # 5. Run Loop
    print(f"\nSTARTING RESCUE FOR {len(bad_sessions)} SESSIONS...")
    stats = {"Success": 0, "Failed": 0}
    
    for sid in bad_sessions:
        print(f"\n>>> Rescuing {sid}...")
        try:
            # EXPLICIT ARGS passed to prevent defaults being None
            res = phase2_pyannote_sota.process_session(
                sid, d, e, anchor, 
                base_dir=PROJECT_ROOT, 
                output_dir=OUTPUT_DIR
            )
            
            if res['status'] == 'SUCCESS':
                print("   SUCCESS!")
                stats["Success"] += 1
            else:
                print(f"   FAILED: {res.get('reason')}")
                stats["Failed"] += 1
        except Exception as ex:
            print(f"   CRASHED: {ex}")
            traceback.print_exc()
            stats["Failed"] += 1
            
    print("\n=== FINAL STATS ===")
    print(stats)

if __name__ == "__main__":
    run_rescue()
