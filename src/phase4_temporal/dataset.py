import os
import torch
from torch.utils.data import Dataset
import glob

import pandas as pd

class TemporalDataset(Dataset):
    def __init__(self, data_dir, labels_path="data/labels/master_labels_v7.csv", split_ids=None, augment=False):
        """
        Args:
            data_dir: Path to directory containing *_fused.pt files
            labels_path: Path to master labels CSV
            split_ids: List of patient IDs to include
            augment: If True, split each sequence into two halves
        """
        self.data_dir = data_dir
        self.augment = augment
        self.files = glob.glob(os.path.join(data_dir, "*_fused.pt"))
        
        # Load Labels
        try:
            self.labels_df = pd.read_csv(labels_path)
            self.labels_df['Participant_ID'] = self.labels_df['Participant_ID'].astype(str)
            self.labels_dict = self.labels_df.set_index('Participant_ID')[['PHQ_Binary', 'PHQ_Score_Norm']].to_dict('index')
        except Exception as e:
            print(f"⚠️ Error loading labels: {e}")
            self.labels_dict = {}

        if split_ids is not None:
            filtered = []
            pid_set = set([str(p) for p in split_ids])
            for f in self.files:
                pid = os.path.basename(f).split('_')[0]
                if pid in pid_set:
                    filtered.append(f)
            self.files = filtered
            print(f"Dataset: Filtered to {len(self.files)} files based on provided IDs.")
        else:
            print(f"Dataset: Loaded all {len(self.files)} files from {data_dir}.")

        if self.augment:
            self.samples = []
            for f in self.files:
                self.samples.append((f, 0))
                self.samples.append((f, 1))
        else:
            self.samples = [(f, None) for f in self.files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, half = self.samples[idx]
        pid = os.path.basename(path).split('_')[0]
        
        # Load features
        data = torch.load(path, map_location='cpu')
        fused = data['fused_embedding'].float()
        
        # Get labels from master_labels_v7
        if pid in self.labels_dict:
            label = torch.tensor(self.labels_dict[pid]['PHQ_Binary'], dtype=torch.float)
            phq_score = torch.tensor(self.labels_dict[pid]['PHQ_Score_Norm'], dtype=torch.float)
        else:
            # Fallback to file's embedded label if missing from CSV
            label = torch.tensor(data.get('label', 0), dtype=torch.float)
            phq_score = torch.tensor(0.0, dtype=torch.float)
        
        if self.augment and half is not None:
            mid = fused.shape[0] // 2
            fused = fused[:mid] if half == 0 else fused[mid:]
            if fused.shape[0] == 0:
                fused = data['fused_embedding'].float()

        return {
            'features': fused,
            'label': label,
            'phq_score': phq_score,
            'id': f"{pid}_{half if half is not None else 'full'}"
        }

def pad_collate(batch):
    """
    Collate function to handle variable length sequences with padding.
    """
    features = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    phq_scores = torch.stack([item['phq_score'] for item in batch])
    ids = [item['id'] for item in batch]
    
    # Pad sequences
    lengths = torch.tensor([f.shape[0] for f in features])
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Create mask (True = Padding)
    max_len = features_padded.shape[1]
    mask = torch.arange(max_len)[None, :] >= lengths[:, None] 
    
    return {
        'features': features_padded, # [B, T_max, 256]
        'mask': mask,                # [B, T_max] (True means padded/ignore)
        'labels': labels,            # [B]
        'phq_scores': phq_scores,    # [B]
        'lengths': lengths           # [B]
    }
