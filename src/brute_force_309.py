
import sys, os, torch
import soundfile as sf
import numpy as np
from pathlib import Path

# Add the pipeline directory to path
sys.path.append('/home/g755389/depressivoice_phase2_outputs')
import phase2_pyannote_sota

def run_brute_force_309():
    print("\n=== BRUTE FORCE RESCUE: Session 309 ===")
    
    # 1. Setup Models
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("ERROR: HF_TOKEN missing!")
        return
        
    d, e = phase2_pyannote_sota.load_models(hf_token)
    
    # 2. Define Paths
    root = Path('/home/g755389/DepressiVoice-LLM')
    audio_path = root / "data/01_raw/309_AUDIO.wav"
    output_dir = Path('/home/g755389/depressivoice_phase2_outputs/session_outputs/309/brute_force')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. FORCE 2 SPEAKERS (AGGRESSIVE CLUSTERING)
    print(f"Diarizing {audio_path.name} with forced 2 speakers...")
    diarization = d(str(audio_path), num_speakers=2)
    
    # 4. Extract Both Speakers
    speakers = diarization.labels()
    print(f"Found {len(speakers)} speaker labels: {speakers}")
    
    for i, spk in enumerate(speakers):
        print(f"Extracting Speaker {spk} to candidate_{i}.wav...")
        
        # Load audio (mono)
        waveform, sr = sf.read(str(audio_path))
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
            
        # Collect segments
        speaker_audio = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            if label == spk:
                start_s = int(segment.start * sr)
                end_s = int(segment.end * sr)
                speaker_audio.append(waveform[start_s:end_s])
        
        if speaker_audio:
            final_audio = np.concatenate(speaker_audio)
            out_file = output_dir / f"candidate_{i}.wav"
            sf.write(str(out_file), final_audio, sr)
            print(f"  SAVED: {out_file} ({len(final_audio)/sr:.1f}s)")
            
    print("\n=== CHECK COMPLETE ===")
    print(f"Please listen to the files in: {output_dir}")
    print("One should be the Patient, the other should be Ellie.")

if __name__ == "__main__":
    run_brute_force_309()
