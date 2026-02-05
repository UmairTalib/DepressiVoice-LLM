import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging

class MultimodalDAICDataset(Dataset):
    """
    Multimodal Dataset for E-DAIC.
    Loads paired Audio and Text embeddings.
    """
    def __init__(self, 
                 audio_dir, 
                 text_dir, 
                 labels_csv, 
                 subset='train',
                 max_audio_len=5000,
                 max_text_len=1000):
        """
        Args:
            audio_dir (str): Path to audio features ({ID}_features_norm.pt).
            text_dir (str): Path to text features ({ID}_text_seq.pt).
            labels_csv (str): Path to detailed_lables.csv.
            subset (str): 'train', 'dev', or 'test'.
            max_audio_len (int): Max sequence length for audio.
            max_text_len (int): Max sequence length for text.
        """
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        # Load Labels
        self.df = pd.read_csv(labels_csv)
        
        # Mapping Participant to participant_id if needed
        # Assuming columns: Participant, Depression_label, split
        if 'Participant' in self.df.columns:
            self.df = self.df.rename(columns={'Participant': 'participant_id', 'Depression_label': 'label'})
            
        self.samples = []
        missing_count = 0
        
        subset_df = self.df[self.df['split'] == subset]
        
        for _, row in subset_df.iterrows():
            pid = int(row['participant_id'])
            label = int(row['label'])
            
            audio_path = os.path.join(audio_dir, f"{pid}_features_norm.pt")
            text_path = os.path.join(text_dir, f"{pid}_text_seq.pt")
            
            if os.path.exists(audio_path) and os.path.exists(text_path):
                self.samples.append({
                    'participant_id': pid,
                    'audio_path': audio_path,
                    'text_path': text_path,
                    'label': label
                })
            else:
                missing_count += 1
                
        print(f"Loaded {len(self.samples)} samples for {subset}. Missing: {missing_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load tensors
        audio_feat = torch.load(sample['audio_path']) # Might be [Windows, SeqLen, 1024]
        text_feat = torch.load(sample['text_path'])   # Might be [Windows, SeqLen, 768]
        
        # Ensure 2D: [T, D]
        if audio_feat.ndim == 3:
            audio_feat = audio_feat.view(-1, audio_feat.shape[-1])
        elif audio_feat.ndim == 1:
            audio_feat = audio_feat.unsqueeze(0) # [1, D] if single vector
            
        if text_feat.ndim == 3:
            text_feat = text_feat.view(-1, text_feat.shape[-1])
        elif text_feat.ndim == 1:
            text_feat = text_feat.unsqueeze(0)
            
        # Truncate to avoid OOM in Cross-Attention
        if audio_feat.shape[0] > self.max_audio_len:
            audio_feat = audio_feat[:self.max_audio_len]
        if text_feat.shape[0] > self.max_text_len:
            text_feat = text_feat[:self.max_text_len]
            
        return {
            'audio_feat': audio_feat,
            'text_feat': text_feat,
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'participant_id': sample['participant_id']
        }

def multimodal_collate_fn(batch):
    """
    Collate function to handle variable length sequences.
    Pads Audio and Text features and creates attention masks.
    """
    audio_feats = [item['audio_feat'] for item in batch]
    text_feats = [item['text_feat'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    pids = [item['participant_id'] for item in batch]
    
    # Pad Audio: [B, T_a_max, 1024]
    audio_padded = pad_sequence(audio_feats, batch_first=True, padding_value=0.0)
    
    # Pad Text: [B, T_t_max, 768]
    text_padded = pad_sequence(text_feats, batch_first=True, padding_value=0.0)
    
    # Create masks (1 for real data, 0 for padding)
    # Audio Mask: [B, T_a_max]
    audio_mask = torch.zeros(audio_padded.shape[0], audio_padded.shape[1], dtype=torch.bool)
    for i, feat in enumerate(audio_feats):
        audio_mask[i, :feat.shape[0]] = 1
        
    # Text Mask: [B, T_t_max]
    text_mask = torch.zeros(text_padded.shape[0], text_padded.shape[1], dtype=torch.bool)
    for i, feat in enumerate(text_feats):
        text_mask[i, :feat.shape[0]] = 1
        
    return {
        'audio_feat': audio_padded,
        'text_feat': text_padded,
        'audio_mask': audio_mask,
        'text_mask': text_mask,
        'label': labels,
        'participant_ids': pids
    }
