import os
import torch
import scipy.io.wavfile as wavfile
import numpy as np
import pandas as pd
from dataset import DepressionAudioDataset
from audiomodel import DepressionAudioModel

def setup_mock_data():
    """Creates a dummy directory structure with 1 fake wav file."""
    os.makedirs("mock_data/100", exist_ok=True)
    
    # Create fake wav using scipy instead of torchaudio to avoid torchcodec issues
    sr = 16000
    duration = 5
    t = np.linspace(0, duration, sr * duration)
    dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t) # A4 note
    
    wavfile.write("mock_data/100/patient_audio.wav", sr, dummy_audio.astype(np.float32))
    
    # Create fake labels csv
    df = pd.DataFrame({'Participant_ID': [100], 'PHQ_Binary': [1], 'split': ['train']})
    df.to_csv("mock_labels.csv", index=False)
    
    print("Created mock data.")

def run_test():
    print("Starting Local Test...")
    setup_mock_data()
    
    # 1. Test Dataset
    ds = DepressionAudioDataset(
        root_dir="mock_data", 
        labels_csv="mock_labels.csv", 
        subset='train',
        target_sr=16000,
        max_duration_sec=8.0
    )
    
    print(f"Dataset Length: {len(ds)}")
    item = ds[0]
    print(f"Sample Keys: {item.keys()}")
    print(f"Input Shape: {item['input_values'].shape}")
    print(f"Mask Shape: {item['attention_mask'].shape}")
    
    # 2. Test Model
    print("Initializing Model...")
    model = DepressionAudioModel(freeze_feature_extractor=True)
    
    # 3. Forward Pass
    print("Running Forward Pass...")
    inputs = item['input_values'].unsqueeze(0) # Batch dim
    mask = item['attention_mask'].unsqueeze(0)
    
    with torch.no_grad():
        logits = model(inputs, attention_mask=mask)
        
    print(f"Logits Shape: {logits.shape}") # Should be (1, 2)
    print(f"Logits: {logits}")
    
    print("✅ TEST PASSED")
    
    # Cleanup
    import shutil
    shutil.rmtree("mock_data")
    os.remove("mock_labels.csv")

if __name__ == "__main__":
    run_test()
