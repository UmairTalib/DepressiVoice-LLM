
import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import logging

class DepressionAudioDataset(Dataset):
    """
    Audio-only dataset for Depression Detection.
    Loads 'patient_audio.wav' (zero-contamination) from valid session folders.
    """
    def __init__(self, 
                 root_dir, 
                 labels_csv, 
                 subset='train', 
                 target_sr=16000, 
                 max_duration_sec=10.0,
                 min_duration_sec=0.5):
        """
        Args:
            root_dir (str): Path to data directory (e.g., 'data/Phase1_results').
                            Structure: root_dir/{participant_id}/patient_audio.wav
            labels_csv (str): Path to CSV containing 'Participant_ID' and 'PHQ_Binary'.
            subset (str): 'train', 'dev', or 'test'. (For splitting logic).
            target_sr (int): Target sampling rate (16000 for WavLM).
            max_duration_sec (float): Max audio chunk length to avoid OOM.
            min_duration_sec (float): Minimum duration to include.
        """
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.max_samples = int(max_duration_sec * target_sr)
        self.min_samples = int(min_duration_sec * target_sr)
        
        # Load Labels
        self.df = pd.read_csv(labels_csv)
        
        # Group by Participant_ID to get one row per session
        # We take the first value for PHQ_Binary since it should be consistent per session
        self.session_df = self.df.groupby('Participant_ID').first().reset_index()
        
        # Filter based on file existence
        self.valid_samples = []
        missing_count = 0
        
        print(f"Initializing {subset} dataset...")
        print(f"Checking root_dir: {self.root_dir}")
        for idx, row in self.session_df.iterrows():
            pid = int(row['Participant_ID'])
            label = int(row['PHQ_Binary'])
            
            # Use 'split' column if it exists
            if 'split' in self.session_df.columns and row['split'] != subset:
                continue
                
            # Check possible variants of the filename
            audio_path_plain = os.path.join(root_dir, str(pid), 'patient_audio.wav')
            audio_path_pid = os.path.join(root_dir, str(pid), f'patient_audio_{pid}.wav')
            
            final_path = None
            if os.path.exists(audio_path_plain):
                final_path = audio_path_plain
            elif os.path.exists(audio_path_pid):
                final_path = audio_path_pid
                
            if idx < 5:
                print(f"DEBUG: Checking path: {audio_path_plain} (exists: {os.path.exists(audio_path_plain)})")
                print(f"DEBUG: Checking path: {audio_path_pid} (exists: {os.path.exists(audio_path_pid)})")
            
            if final_path:
                self.valid_samples.append({
                    'participant_id': pid,
                    'audio_path': final_path,
                    'label': label
                })
            else:
                missing_count += 1
        
        print(f"Dataset Loaded: {len(self.valid_samples)} valid samples. ({missing_count} missing)")
        
        # Calculate class weights for imbalance handling
        labels = [s['label'] for s in self.valid_samples]
        if len(labels) > 0:
            num_0 = labels.count(0)
            num_1 = labels.count(1)
            total = len(labels)
            print(f"Class Distribution: Non-Depressed (0): {num_0}, Depressed (1): {num_1}")
            # Weight for class 1 = (Total / 2) / Count_1
            self.pos_weight = (total / 2.0) / num_1 if num_1 > 0 else 1.0
            print(f"Calculated Positive Weight: {self.pos_weight:.2f}")

    def __len__(self):
        return len(self.valid_samples)

    def _load_audio(self, audio_path):
        """Robust audio loading with torchaudio primary and scipy/numpy fallback."""
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
            return waveform, self.target_sr
        except Exception as e:
            # Fallback to scipy for environments with broken torchaudio backends
            import scipy.io.wavfile as wavfile
            import numpy as np
            sr, data = wavfile.read(audio_path)
            # Convert to torch tensor
            waveform = torch.from_numpy(data.astype(np.float32))
            # Ensure (Channels, Time)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1) # (T, C) -> (C, T)
            
            # Resample if needed
            if sr != self.target_sr:
                # Basic linear resampling via numpy if torchaudio resample is risky
                num_samples = int(len(data) * self.target_sr / sr)
                data_resampled = np.interp(
                    np.linspace(0, len(data), num_samples),
                    np.arange(len(data)),
                    data
                )
                waveform = torch.from_numpy(data_resampled.astype(np.float32)).unsqueeze(0)
            
            return waveform, self.target_sr

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        audio_path = sample['audio_path']
        label = sample['label']
        pid = sample['participant_id']
        
        # Load Audio (Robust)
        waveform, sr = self._load_audio(audio_path)
        
        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Squeeze channel dim: (1, T) -> (T,)
        waveform = waveform.squeeze(0)
        
        # Windowing Strategy for Training
        # Ideally, we should train on random chunks or processed chunks.
        # For simplicity in this scaffold: Random Crop if > max_len, Pad if < max_len
        # (WavLM can handle variable lengths, but padding helps batching)
        
        length = waveform.shape[0]
        
        if length > self.max_samples:
            # Random Crop for training
            start = torch.randint(0, length - self.max_samples, (1,)).item()
            waveform = waveform[start : start + self.max_samples]
        elif length < self.max_samples:
            # Pad
            padding = self.max_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Attention Mask (1 for real data, 0 for padding)
        # Note: WavLM expects 'attention_mask' if padding is used?
        # HuggingFace WavLM processor creates mask. We can create it manually here.
        # Mask: 1 for valid samples
        attention_mask = torch.ones(self.max_samples)
        if length < self.max_samples:
             attention_mask[length:] = 0
             
        return {
            'input_values': waveform, # (T,)
            'attention_mask': attention_mask, # (T,)
            'labels': torch.tensor(label, dtype=torch.long),
            'participant_id': str(pid)
        }

if __name__ == "__main__":
    # Test Block
    print("Testing dataset loader...")
    # Mocking a csv for test would be needed or path to existing one
