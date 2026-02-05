import json
import scipy.io.wavfile as wav
import os

def verify_session(sid):
    json_path = f"data/processed/phase1/session_outputs/{sid}/cleaned_patient_text_v2.json"
    wav_path = f"data/repaired_audio/patient_audio_{sid}.wav"
    
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    sr, data = wav.read(wav_path)
    actual_duration = len(data) / sr
    
    expected_duration = sum(s['end'] - s['start'] for s in meta['segments'])
    
    print(f"Session {sid}:")
    print(f"  - Metadata expected speech: {expected_duration:.2f}s")
    print(f"  - Actual WAV speech length: {actual_duration:.2f}s")
    print(f"  - Difference: {abs(actual_duration - expected_duration):.4f}s")
    
    if abs(actual_duration - expected_duration) < 0.01:
        print("✅ SUCCESS: Audio perfectly matches clean metadata.")
    else:
        print("❌ ERROR: Duration mismatch!")

if __name__ == "__main__":
    verify_session("300")
    verify_session("301")
