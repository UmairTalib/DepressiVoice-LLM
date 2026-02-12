import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
import glob
from typing import List, Dict, Optional, Tuple

class DepressionDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train',
                 window_size: int = 8,
                 stride: int = 2,
                 scalars_scaler = None,
                 labels_csv: Optional[str] = None):
        """
        Args:
            root_dir (str): Project root directory.
            split (str): 'train', 'dev', or 'test'.
            window_size (int): Number of rows per window (default 8).
            stride (int): Stride for windowing (default 2).
            scalars_scaler: Fitted scaler (e.g. StandardScaler) for scalars. If None, logic to fit one should be handled externally or strictly for 'train' split if implementing self-fit. 
                            However, usually we pass a fitted scaler from the training set to dev/test.
        """
        self.root_dir = root_dir
        self.split = split
        self.window_size = window_size
        self.stride = stride
        
        # Paths
        if labels_csv is None:
            self.labels_path = os.path.join(root_dir, 'data/01_raw/detailed_lables.csv')
        else:
            self.labels_path = labels_csv
        self.audio_dir = os.path.join(root_dir, 'data/phase2_clean/audio_embeddings_normalized')
        self.text_dir = os.path.join(root_dir, 'data/phase2_clean/text_embeddings_clean')
        self.transcripts_dir = os.path.join(root_dir, 'phase1_rerun_outputs') # Assuming subdirs like session_{ID}/transcript.json

        # Load Labels and Filter by Split
        self.metadata = self._load_metadata()
        
        # Load Scalars (and fit scaler if needed/provided)
        self.scalars_scaler = scalars_scaler
        self.patient_scalars = self._load_all_scalars()

        # Load PHQ-8 Phenotypes (Symptoms and Severity)
        self.patient_phenotypes = self._load_phenotypes()

        # Prepare Windows
        self.windows = self._create_windows()
        
        print(f"Dataset initialized for split '{split}' with {len(self.windows)} windows from {len(self.metadata)} patients.")

    def _load_metadata(self) -> pd.DataFrame:
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found at {self.labels_path}")
        
        df = pd.read_csv(self.labels_path)
        # Filter by split
        df = df[df['split'] == self.split].reset_index(drop=True)
        return df

    def _load_all_scalars(self) -> Dict[str, torch.Tensor]:
        """
        Loads scalars for all patients in the split.
        Scalars: Mean Response Latency, Voice Activity Ratio, Articulation Rate.
        """
        raw_scalars = []
        patient_ids = []

        for idx, row in self.metadata.iterrows():
            pid = str(row['Participant'])
            
            # Check masked directory first, then base
            masked_path = os.path.join(self.root_dir, "phase1_rerun_outputs_masked", f"session_{pid}", "transcript.json")
            base_path = os.path.join(self.root_dir, "phase1_rerun_outputs", f"session_{pid}", "transcript.json")
            transcript_path = masked_path if os.path.exists(masked_path) else base_path
            
            # Default values if missing (should ideally alert user)
            lat, var, ar = 0.0, 0.0, 0.0
            
            if os.path.exists(transcript_path):
                try:
                    with open(transcript_path, 'r') as f:
                        data = json.load(f)
                        segments = data.get('segments', [])
                        
                        if not segments:
                            lat, var, ar = 0.0, 0.0, 0.0
                        else:
                            # 1. Voice Activity Ratio (VAR)
                            # Total patient speech duration
                            speech_durations = [seg['end'] - seg['start'] for seg in segments]
                            total_speech = sum(speech_durations)
                            
                            # Total session duration (from start of first to end of last)
                            # Or from 0? Let's use end of last segment as session approx if no metadata
                            session_end = segments[-1]['end']
                            session_start = segments[0]['start']
                            session_duration = session_end - session_start
                            
                            if session_duration > 0:
                                var = total_speech / session_duration
                            else:
                                var = 0.0
                                
                            # 2. Mean Pause Duration
                            # Gaps between segments
                            if len(segments) > 1:
                                pauses = []
                                for i in range(1, len(segments)):
                                    gap = segments[i]['start'] - segments[i-1]['end']
                                    if gap > 0: # Only positive gaps
                                        pauses.append(gap)
                                if pauses:
                                    lat = sum(pauses) / len(pauses)
                                else:
                                    lat = 0.0
                            else:
                                lat = 0.0 # No pauses if 0 or 1 segment
                                
                            # 3. Articulation Rate (Words per second of speech)
                            # Count words in 'text' field
                            total_words = 0
                            for seg in segments:
                                text = seg.get('text', "")
                                total_words += len(text.split())
                                
                            if total_speech > 0:
                                ar = total_words / total_speech
                            else:
                                ar = 0.0

                except Exception as e:
                    print(f"Error loading {transcript_path}: {e}")
                    lat, var, ar = 0.0, 0.0, 0.0
            
            raw_scalars.append([lat, var, ar])
            patient_ids.append(pid)
            
        # Normalize
        raw_tensor = torch.tensor(raw_scalars, dtype=torch.float32)
        
        if self.scalars_scaler is None and self.split == 'train':
            # Fit mean and std on the training split
            mean = raw_tensor.mean(dim=0, keepdim=True)
            std = raw_tensor.std(dim=0, keepdim=True) + 1e-6
            self.scalars_scaler = (mean, std)
            print(f"Scalars Scaler fitted on Train: Mean={mean.tolist()}, Std={std.tolist()}")
                 
        if self.scalars_scaler is not None:
            mean, std = self.scalars_scaler
            raw_tensor = (raw_tensor - mean) / std
            
        # Map back to ID
        scalar_map = {pid: raw_tensor[i] for i, pid in enumerate(patient_ids)}
        return scalar_map

    def _load_phenotypes(self) -> Dict[str, Dict]:
        """
        Loads individual PHQ-8 symptoms and total severity.
        """
        phenotypes = {}
        symptom_cols = [f'PHQ8_{i}_{name}' for i, name in enumerate([
            'NoInterest', 'Depressed', 'Sleep', 'Tired', 'Appetite', 'Failure', 'Concentration', 'Psychomotor'
        ], 1)]
        
        for _, row in self.metadata.iterrows():
            pid = str(row['Participant'])
            symptoms = row[symptom_cols].values.astype(float) / 3.0 # Scale to 0-1
            severity = float(row['Depression_severity']) / 24.0   # Scale to 0-1
            
            phenotypes[pid] = {
                'symptoms': torch.tensor(symptoms, dtype=torch.float),
                'severity': torch.tensor(severity, dtype=torch.float)
            }
        return phenotypes

    def _create_windows(self) -> List[Tuple[str, int, int]]:
        """
        Slices sequences into windows.
        Returns list of (patient_id, start_idx, label).
        """
        windows = []
        for idx, row in self.metadata.iterrows():
            pid = str(row['Participant'])
            label = int(row['Depression_label']) # Binary label
            
            # Check files exist
            audio_path = os.path.join(self.audio_dir, f"{pid}_features.pt")
            text_path = os.path.join(self.text_dir, f"{pid}_features.pt")
            
            if not os.path.exists(audio_path) or not os.path.exists(text_path):
                # print(f"Missing data for {pid}, skipping.")
                continue
                
            # We assume perfect alignment, so we can check length of one
            # To be fast, maybe we don't load the whole tensor here? 
            # But we need length. loading on cluster might be fast enough.
            # Only way to know T is to load.
            try:
                # We interpret "features.pt" as torch tensors
                # Ideally we cache lengths to avoid reloading every time on init.
                # For this implementation, we load.
                # Optimization: Could just read header or shape if possible.
                # But PyTorch load is usually whole file.
                audio_t = torch.load(audio_path, map_location='cpu')
                T = audio_t.shape[0]
                
                num_windows = (T - self.window_size) // self.stride + 1
                for w in range(num_windows):
                    start = w * self.stride
                    windows.append((pid, start, label))
                    
            except Exception as e:
                print(f"Error processing {pid}: {e}")
                
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        pid, start, label = self.windows[idx]
        
        # Load files (Caching could be used here for optimization)
        # Re-loading for every window is IO heavy. 
        # Better strategy: Data loader uses workers, OS caches files.
        audio_path = os.path.join(self.audio_dir, f"{pid}_features.pt")
        text_path = os.path.join(self.text_dir, f"{pid}_features.pt")
        
        audio_full = torch.load(audio_path, map_location='cpu').float() # [T, Frames, 1024] or [T, 1024]
        text_full = torch.load(text_path, map_location='cpu').float()   # [T, 768]
        
        # Handle 3D Audio (Segment-Aligned with padding): [T, Frames, D] -> [T, D]
        if audio_full.dim() == 3:
            # Check if it's singleton (squeeze) or actual frames (mean pool)
            if audio_full.shape[1] == 1:
                audio_full = audio_full.squeeze(1)
            else:
                audio_full = audio_full.mean(dim=1)
            
        # Handle 3D Text if singleton
        if text_full.dim() == 3 and text_full.shape[1] == 1:
            text_full = text_full.squeeze(1)
        
        # Slice
        end = start + self.window_size
        audio_win = audio_full[start:end]
        text_win = text_full[start:end]
        
        # Sanitization: Force Audio[t] = 0.0 if Text[t] is "masked"
        # Logic: If text vector is all zeros (common for mask) or specific token?
        # Assuming mask vector is all zeros for now based on "Clean (Interviewer masked)"
        # Check if text row is zero vector
        is_silence = torch.all(torch.abs(text_win) < 1e-6, dim=1) # Boolean mask [window_size]
        
        # Apply mask to audio
        # audio_win[is_silence] = 0.0 # This modifies in place? Safe since we loaded fresh or sliced
        # Safe way:
        audio_win = audio_win.clone()
        audio_win[is_silence] = 0.0
        
        # Get Scalars
        scalars = self.patient_scalars.get(pid, torch.zeros(3)) # Default if missing
        
        # Get PHQ-8 Phenotypes
        pheno = self.patient_phenotypes.get(pid, {
            'symptoms': torch.zeros(8),
            'severity': torch.tensor(0.0)
        })
        
        return {
            'audio': audio_win,      # [8, 1024]
            'text': text_win,        # [8, 768]
            'scalars': scalars,      # [3]
            'label': torch.tensor(label, dtype=torch.float), # Binary
            'severity': pheno['severity'],
            'symptoms': pheno['symptoms'],
            'id': pid
        }

if __name__ == "__main__":
    # Sanity check
    print("Dataset module defined.")
