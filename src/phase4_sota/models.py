import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [Batch, Seq, Dim]
        return x + self.pe[:, :x.size(1), :]

class CrossModalTransformerAggregator(nn.Module):
    """
    Phase 4 Upgrade: Session-level Cross-Modal Aggregator.
    Captures global context and boosts precision through Bi-directional Co-Attention.
    Input: [Batch, Time, 256] (Fused embeddings from Phase 3.5)
    """
    def __init__(self, input_dim=256, d_model=256, nhead=8, num_layers=2, dropout=0.3, temperature=1.0):
        super(CrossModalTransformerAggregator, self).__init__()
        self.temperature = temperature
        self.d_model = d_model
        
        # 1. Dimension Projection (Splitting/Projecting fused features into dual streams)
        self.audio_proj = nn.Linear(input_dim, d_model)
        self.text_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Cross-Modal Attention Layers
        # We use MultiheadAttention for Co-Attention: 
        # Stream A (Audio Query) attends to Stream B (Text Key/Value)
        # Stream B (Text Query) attends to Stream A (Audio Key/Value)
        self.audio_co_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.text_co_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norms_a = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms_t = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # 3. Final Fusion & Global Aggregation
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.norm_final = nn.LayerNorm(d_model)
        
        # Head 1: Classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Head 2: PHQ-8 Severity Regression
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask=None):
        """
        x: [Batch, Time, 256]
        mask: [Batch, Time] (True means PAD step)
        """
        # A. Projection & Positional Encoding
        a_stream = self.dropout(self.pos_encoder(self.audio_proj(x)))
        t_stream = self.dropout(self.pos_encoder(self.text_proj(x)))
        
        # B. Bi-directional Co-Attention
        attn_list = []
        for i in range(len(self.audio_co_attn)):
            # Audio Query attending to Text Key/Value
            res_a, w_a = self.audio_co_attn[i](query=a_stream, key=t_stream, value=t_stream, key_padding_mask=mask)
            a_stream = self.norms_a[i](a_stream + res_a)
            
            # Text Query attending to Audio Key/Value
            res_t, w_t = self.text_co_attn[i](query=t_stream, key=a_stream, value=a_stream, key_padding_mask=mask)
            t_stream = self.norms_t[i](t_stream + res_t)
            
            if w_a is not None and w_t is not None:
                attn_list.append((w_a + w_t) / 2.0)
            
        # C. Global Aggregation
        fused = torch.cat([a_stream, t_stream], dim=-1) # [Batch, Time, 2*d_model]
        fused = self.norm_final(self.fusion_layer(fused))
        
        # Global Average Pooling (respecting mask)
        if mask is not None:
            # mask: 1 for pad, 0 for valid
            numeric_mask = (~mask).float().unsqueeze(-1)
            sum_fused = (fused * numeric_mask).sum(dim=1)
            count_valid = numeric_mask.sum(dim=1).clamp(min=1e-6)
            global_embedding = sum_fused / count_valid
        else:
            global_embedding = fused.mean(dim=1)
            
        # D. Classification & Severity Regression
        logits = self.classifier(global_embedding)
        logits = logits / self.temperature
        
        severity = self.regression_head(global_embedding)
        
        # Return rollout attention (mean across layers)
        rollout_attn = torch.stack(attn_list).mean(dim=0) if attn_list else None
        
        return (logits, severity), rollout_attn 

class ImprovedLSTMAggregator(nn.Module):
    """
    Phase 4 Hybrid Aggregator: Bi-LSTM with Dual Pooling.
    """
    def __init__(self, input_dim=256, hidden_dim=64, num_layers=1, dropout=0.5, temperature=1.0):
        super(ImprovedLSTMAggregator, self).__init__()
        self.temperature = temperature
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, mask=None):
        outputs, _ = self.lstm(x)
        attn_scores = self.attention(outputs)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask_expanded, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(outputs * attn_weights, dim=1)
        if mask is not None:
            outputs_for_max = outputs.masked_fill(mask_expanded, -1e9)
            peak_vector, _ = torch.max(outputs_for_max, dim=1)
        else:
            peak_vector, _ = torch.max(outputs, dim=1)
        final_embedding = torch.cat([context_vector, peak_vector], dim=1)
        logits = self.classifier(final_embedding)
        logits = logits / self.temperature
        
        # MTL Fallback: Return dummy severity
        return (logits, torch.zeros_like(logits)), attn_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_weight * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()
