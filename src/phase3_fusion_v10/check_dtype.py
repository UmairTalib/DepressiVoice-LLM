import torch
import os

audio_path = os.path.expanduser('~/DepressiVoice-LLM/data/phase2_clean/audio_embeddings_normalized/300_features.pt')
text_path = os.path.expanduser('~/DepressiVoice-LLM/data/phase2_clean/text_embeddings_clean/300_features.pt')

try:
    a = torch.load(audio_path, map_location='cpu')
    print(f"Audio dtype: {a.dtype}, Shape: {a.shape}")
except Exception as e:
    print(f"Audio load failed: {e}")

try:
    t = torch.load(text_path, map_location='cpu')
    print(f"Text dtype: {t.dtype}, Shape: {t.shape}")
except Exception as e:
    print(f"Text load failed: {e}")
