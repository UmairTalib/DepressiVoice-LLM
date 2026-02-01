
import sys, os, torch, json
from pathlib import Path
from tqdm import tqdm

# Add the pipeline directory to path
sys.path.append('/home/g755389/depressivoice_phase2_outputs')
try:
    import phase2_pyannote_sota
except ImportError:
    print("Error: Could not import phase2_pyannote_sota. Make sure paths are correct.")
    sys.exit(1)

def generate_roles_gpu():
    print("\n=== GPU ROLE METADATA GENERATION (FIXED PATHS) ===")
    
    # 1. Setup Environment
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is missing!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Models
    print("Loading Speaker Verification models...")
    d, e = phase2_pyannote_sota.load_models(hf_token)
    
    # 3. Setup Paths
    root = Path('/home/g755389/DepressiVoice-LLM')
    output_base = Path('/home/g755389/depressivoice_phase2_outputs/session_outputs')
    
    # 4. Create Ellie Anchor
    print("Generating Interviewer Biometric Anchor...")
    wavs = sorted(root.glob('data/01_raw/*_AUDIO.wav'))
    anchor = phase2_pyannote_sota.create_ellie_anchor(wavs, d, e)
    
    # 5. Process Sessions
    session_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()])
    print(f"Found {len(session_dirs)} session folders.")
    
    count = 0
    skipped = 0
    missing_files = 0
    
    for sdir in tqdm(session_dirs, desc="Generating Roles"):
        roles_file = sdir / 'roles.json'
        
        # Skip if already has full metadata
        if roles_file.exists():
            skipped += 1
            continue
            
        sid = sdir.name
        
        # FIXED: Check for both [SID].rttm and diarization.rttm
        rttm_variants = [sdir / f"{sid}.rttm", sdir / "diarization.rttm"]
        rttm_file = next((r for r in rttm_variants if r.exists()), None)
        
        # Audio file check
        audio_file = root / "data/01_raw" / f"{sid}_AUDIO.wav"
        
        if not rttm_file or not audio_file.exists():
            missing_files += 1
            continue
            
        try:
            # Load diarization from RTTM
            from pyannote.core import Annotation, Segment
            diarization = Annotation(uri=sid)
            with open(rttm_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 8: continue
                    start = float(parts[3])
                    duration = float(parts[4])
                    label = parts[7]
                    diarization[Segment(start, start + duration), '_'] = label
            
            # Use established role logic
            roles, similarities, m, c = phase2_pyannote_sota.assign_roles(
                sid, diarization, str(audio_file), e, anchor
            )
            
            # Save the "Evidence"
            with open(roles_file, 'w') as f:
                json.dump({
                    "roles": roles,
                    "similarities": similarities,
                    "best_ellie_match": m,
                    "anchor_confidence": c
                }, f, indent=2)
            count += 1
        except Exception as ex:
            print(f"\n[ERROR] Failed session {sid}: {ex}")

    print(f"\n=== GENERATION COMPLETE ===")
    print(f"New roles.json created: {count}")
    print(f"Skipped (Already exists): {skipped}")
    print(f"Missing Files (RTTM/WAV): {missing_files}")
    print(f"Total Folders Scanned: {len(session_dirs)}")

if __name__ == "__main__":
    generate_roles_gpu()
