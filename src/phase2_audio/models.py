
import torch
import torch.nn as nn
from transformers import AutoModel, WavLMModel

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        # Q=Text, K=Audio, V=Audio
        attn_output, _ = self.attn(query, key, value)
        return self.norm(query + attn_output)

class MultimodalDepressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-trained Encoders
        self.text_encoder = AutoModel.from_pretrained("mental/mental-roberta-base")
        self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-large")
        
        # Audio Projector (1024 -> 768 to match RoBERTa)
        self.audio_proj = nn.Linear(1024, 768)
        
        # Fusion
        self.fusion = CrossAttentionLayer(d_model=768)
        
        # Regression Head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1) # PHQ-8 Score
        )

    def forward(self, input_ids, attention_mask, audio_values):
        # Text Stream (Batch, Seq, 768)
        text_out = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Audio Stream (Batch, Frames, 1024) -> Project to 768
        # Extract middle layers (TODO: hook implementation)
        audio_out = self.audio_encoder(audio_values).last_hidden_state
        audio_out = self.audio_proj(audio_out)
        
        # Fusion (Text queries Audio)
        fused = self.fusion(text_out, audio_out, audio_out)
        
        # Pool (CLS)
        pooled = fused[:, 0, :]
        
        return self.classifier(pooled)
