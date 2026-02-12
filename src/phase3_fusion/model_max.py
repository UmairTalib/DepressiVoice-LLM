import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalContextBlock(nn.Module):
    """
    Mini-Temporal head using 1D convolution to capture context 
    across local window sequences.
    """
    def __init__(self, d_model):
        super(TemporalContextBlock, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        x_t = x.transpose(1, 2) # [B, D, T]
        out = self.conv(x_t).transpose(1, 2) # [B, T, D]
        return self.norm(x + out) # Residual

class GatedScalarUnit(nn.Module):
    """
    Dynamic integration for clinician-derived scalars.
    Instead of adding them, the model decides if scalars should override deep features.
    """
    def __init__(self, d_model, d_scalar):
        super(GatedScalarUnit, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model + d_scalar, d_model),
            nn.Sigmoid()
        )
        self.scalar_proj = nn.Linear(d_scalar, d_model)
        
    def forward(self, x, s):
        # x: [B, D]
        # s: [B, d_scalar]
        combined = torch.cat([x, s], dim=-1)
        g = self.gate(combined)
        s_feat = torch.relu(self.scalar_proj(s))
        return (1 - g) * x + g * s_feat

class FusionModelMax(nn.Module):
    """
    Phase 3.5 MAX ARCHITECTURE
    - Orthogonal Modality streams
    - Bidirectional Co-Attention
    - Gated Multimodal Unit (GMU)
    - Temporal context aggregation
    - Gated Clinical Scalars
    """
    def __init__(self, 
                 d_audio=1024, 
                 d_text=768, 
                 d_model=256, 
                 nhead=8, 
                 num_layers=2,
                 dropout=0.3):
        super(FusionModelMax, self).__init__()
        
        # 1. Adaptors
        self.audio_proj = nn.Linear(d_audio, d_model)
        self.text_proj = nn.Linear(d_text, d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # 2. Bidirectional Co-Attention (Parallel Querying)
        self.audio_query_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.text_query_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 3. Gated Multimodal Unit (GMU)
        self.gmu_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 4. Temporal Context Head
        self.temporal_head = TemporalContextBlock(d_model)
        
        # 5. Gated Scalar Integration (Latency, VAR, ART)
        self.scalar_gate = GatedScalarUnit(d_model, d_scalar=3)
        
        # 6. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # 7. MTL Heads (Regression)
        self.severity_head = nn.Linear(d_model, 1)
        self.symptom_head = nn.Linear(d_model, 8)

    def forward(self, audio, text, scalars, audio_mask=None, text_mask=None):
        # ensure dtypes
        audio = audio.to(dtype=self.audio_proj.weight.dtype)
        text = text.to(dtype=self.text_proj.weight.dtype)
        scalars = scalars.to(dtype=self.audio_proj.weight.dtype)

        # Projections
        a_feat = torch.relu(self.norm_a(self.audio_proj(audio)))
        t_feat = torch.relu(self.norm_t(self.text_proj(text)))
        
        # Branch 1: Audio attends to Text
        # key_padding_mask: True to MASK, so ~text_mask
        ma = ~text_mask if text_mask is not None else None
        h_a, _ = self.audio_query_attn(query=a_feat, key=t_feat, value=t_feat, key_padding_mask=ma)
        
        # Branch 2: Text attends to Audio
        mt = ~audio_mask if audio_mask is not None else None
        h_t, _ = self.text_query_attn(query=t_feat, key=a_feat, value=a_feat, key_padding_mask=mt)
        
        # --- GMU Fusion ---
        z = self.gmu_gate(torch.cat([h_a, h_t], dim=-1))
        fused = z * h_a + (1 - z) * h_t
        
        # --- Temporal Context aggregation ---
        fused_context = self.temporal_head(fused)
        
        # --- Pooling ---
        if text_mask is not None:
            m_exp = text_mask.unsqueeze(-1).float()
            pooled = (fused_context * m_exp).sum(dim=1) / m_exp.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = fused_context.mean(dim=1)
            
        # --- Gated Scalar Integration ---
        final_repr = self.scalar_gate(pooled, scalars)
        
        logits = self.classifier(final_repr)
        
        # --- MTL Predictions ---
        sev_pred = self.severity_head(final_repr)
        sym_pred = self.symptom_head(final_repr)
        
        # Return components for Orthogonal Loss Calculation
        return {
            'logits': logits,
            'severity': sev_pred,
            'symptoms': sym_pred,
            'h_audio': h_a,
            'h_text': h_t,
            'z_gate': z
        }
