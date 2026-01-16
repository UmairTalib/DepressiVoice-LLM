import os
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import shutil
import sys

def create_dummy_data():
    os.makedirs("data/01_raw", exist_ok=True)
    os.makedirs("data/02_transcribed", exist_ok=True)

    # Create dummy manifest
    manifest_data = {
        'Participant_ID': ['test_participant'],
        'PHQ8_Binary': [0],
        'PHQ8_Score': [5],
        'Gender': [1]
    }
    pd.DataFrame(manifest_data).to_csv("data/01_raw/metadata_mapped.csv", index=False)
    print("Created dummy manifest.")

    # Create dummy audio (1 second sine wave)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav.write("data/01_raw/test_participant_AUDIO.wav", sample_rate, (audio * 32767).astype(np.int16))
    print("Created dummy audio.")

def check_results():
    output_path = "data/02_transcribed/transcripts_whisperx.csv"
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print("Transcripts generated successfully.")
        print("Columns:", df.columns.tolist())
        expected_cols = ['start_time', 'end_time', 'transcript', 'speaker']
        if all(col in df.columns for col in expected_cols):
            print("SUCCESS: Timestamps and Speaker columns found.")
            print(df[['Participant_ID', 'start_time', 'end_time', 'speaker', 'transcript']].head())
        else:
            print(f"FAILURE: Missing columns. Found: {df.columns.tolist()}")
    else:
        print("FAILURE: Output file not found.")

if __name__ == "__main__":
    create_dummy_data()
    try:
        # Import main only here to allow dummy data creation even if imports fail
        from src.data_processing import main
        main()
        check_results()
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure requirements.txt is installed.")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
