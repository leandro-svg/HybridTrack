import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class DetectionContextEncoder(nn.Module):
    """
    Enhanced encoder for per-frame detections [x,y,z,l,w,h,ry,score,dvx,dvy,dvz] -> embeddings (B, N, D).
    Supports key_padding_mask for variable-length via attention layers that consume masks.
    
    Input format: [x, y, z, l, w, h, ry, score, dvx, dvy, dvz] (11-dim with velocities)
    - x, y, z: 3D coordinates in KITTI velodyne frame (typically -50 to 100m)
    - l, w, h: dimensions (typically 0.5 to 5m)
    - ry: rotation angle (-pi to pi)
    - score: detection confidence (0 to 1)
    - dvx, dvy, dvz: relative velocity (m/s) - optional, may be 0 if unavailable
    
    Also supports legacy 8-dim format without velocities for backward compatibility.
    
    ENHANCED: Adds spatial reasoning features (distance, angle) and deeper encoding.
    """
    def __init__(self, in_dim: int = 8, embed_dim: int = 64, use_dropout: bool = True):
        super().__init__()
        
        # Spatial feature extraction
        self.spatial_features_dim = 3  # distance, angle, height_diff
        # Input will be padded to 11 (8+3 velocity), then +3 spatial = 14 total
        self.base_input_dim = 11  # After velocity padding
        extended_dim = self.base_input_dim + self.spatial_features_dim  # 14 total
        
        # Learned input normalization - learns appropriate scales during training
        self.input_norm = nn.LayerNorm(extended_dim, elementwise_affine=True)
        
        # Deeper encoding network (3-layer instead of 2)
        self.proj1 = nn.Linear(extended_dim, embed_dim * 2)
        self.proj2 = nn.Linear(embed_dim * 2, embed_dim)
        self.proj3 = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.1)
            self.dropout3 = nn.Dropout(0.1)
        
        # Initialize with moderate weights
        nn.init.xavier_uniform_(self.proj1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj3.weight, gain=0.5)
        nn.init.zeros_(self.proj1.bias)
        nn.init.zeros_(self.proj2.bias)
        nn.init.zeros_(self.proj3.bias)

    def _add_spatial_features(self, det_context: torch.Tensor) -> torch.Tensor:
        """
        Add spatial reasoning features: distance, angle, height difference.
        
        Args:
            det_context: (B, N, 8/11) with [dx, dy, dz, ...]
        Returns:
            (B, N, 8/11 + 3) with additional spatial features
        """
        B, N, D = det_context.shape
        
        # Extract relative positions
        dx = det_context[:, :, 0]  # (B, N)
        dy = det_context[:, :, 1]  # (B, N)
        dz = det_context[:, :, 2]  # (B, N)
        
        # Compute spatial features
        distance = torch.sqrt(dx**2 + dy**2 + 1e-8)  # (B, N) horizontal distance
        angle = torch.atan2(dy, dx)  # (B, N) angle in [-pi, pi]
        height_diff = dz  # (B, N) vertical separation
        
        # Stack spatial features
        spatial = torch.stack([distance, angle, height_diff], dim=-1)  # (B, N, 3)
        
        # Concatenate with original features
        return torch.cat([det_context, spatial], dim=-1)  # (B, N, D+3)
    
    def forward(self, det_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            det_context: (B, N, 8) or (B, N, 11) detection features. 
                        8-dim: [dx, dy, dz, l, w, h, ry, score]
                        11-dim: [dx, dy, dz, l, w, h, ry, score, dvx, dvy, dvz]
                        Padded rows may be zeros, masked in attention layer.
        Returns:
            (B, N, embed_dim) embeddings
        """
        # Handle empty input
        if det_context.numel() == 0:
            B = det_context.shape[0] if det_context.dim() >= 1 else 1
            return torch.zeros(B, 0, self.proj3.out_features, 
                             device=det_context.device, dtype=det_context.dtype)
        
        # Handle both 8-dim (training/baseline) and 11-dim (future with velocities) inputs
        if det_context.shape[-1] == 8:
            # Pad 8-dim to 11-dim with zero velocities for model compatibility
            B, N, _ = det_context.shape
            zero_vel = torch.zeros(B, N, 3, device=det_context.device, dtype=det_context.dtype)
            det_context = torch.cat([det_context, zero_vel], dim=-1)  # (B, N, 11)
        
        # ENHANCED: Add spatial reasoning features
        det_context = self._add_spatial_features(det_context)  # (B, N, 11+3=14)
        
        # Learned normalization handles scale differences
        x = self.input_norm(det_context)
        
        # Three-layer projection with GELU and optional dropout
        x = self.proj1(x)
        x = F.gelu(x)
        if self.use_dropout:
            x = self.dropout1(x)
        
        x = self.proj2(x)
        x = F.gelu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        
        x = self.proj3(x)
        x = F.gelu(x)
        if self.use_dropout:
            x = self.dropout3(x)
        
        x = self.output_norm(x)
        
        return x


class HistoryContextEncoder(nn.Module):
    """
    Encodes other agents' histories of length H with state dim=7 -> (B, K, D).
    Input: (B, K, H, 7), key_padding_mask over K handled in attention, not here.
    """
    def __init__(self, state_dim: int = 7, embed_dim: int = 64, use_transformer: bool = True, nhead: int = 4):
        super().__init__()
        self.use_transformer = use_transformer
        self.input_proj = nn.Linear(state_dim, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len=64)
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.temporal_encoder = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, hist_context: torch.Tensor, hist_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hist_context: (B, K, H, 7)
        B, K, H, D = hist_context.shape
        x = hist_context.reshape(B * K, H, D)          # (BK, H, 7)
        x = self.input_proj(x)                          # (BK, H, E)
        x = self.pos(x)
        if self.use_transformer:
            # No padding along time assumed; if needed, extend with time masks
            x = self.temporal_encoder(x)               # (BK, H, E)
        else:
            x, _ = self.temporal_encoder(x)            # (BK, H, E)
        # Pool over time
        x = x.transpose(1, 2)                          # (BK, E, H)
        x = self.pool(x).squeeze(-1)                   # (BK, E)
        x = x.reshape(B, K, -1)                        # (B, K, E)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Multi-head attention from ego query (B, 1, E) to context keys/values (B, L, E)
    with key_padding_mask (B, L) where True masks out positions.
    Returns fused ego embedding (B, E).
    
    Handles edge cases:
    - All positions masked: returns zeros
    - Empty context: returns zeros
    """
    def __init__(self, embed_dim: int = 64, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, dropout=dropout)
        # Temperature parameter for attention scaling - helps with gradient flow
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)  # Learnable temperature

    def forward(self, ego_q: torch.Tensor, ctx_kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            ego_q: (B, 1, E) query from ego vehicle
            ctx_kv: (B, L, E) context key/values
            key_padding_mask: (B, L) True for positions to mask out
        Returns:
            (B, E) fused embedding
        """
        B = ego_q.shape[0]
        
        # Handle empty context
        if ctx_kv.shape[1] == 0:
            return torch.zeros(B, self.embed_dim, device=ego_q.device, dtype=ego_q.dtype)
        
        # Handle all-masked case: if all positions are masked, attention would be undefined
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)  # (B,)
            if all_masked.any():
                # For samples with all masked, we'll handle separately
                # Temporarily unmask at least one position to avoid NaN
                safe_mask = key_padding_mask.clone()
                safe_mask[all_masked, 0] = False  # Unmask first position
            else:
                safe_mask = key_padding_mask
        else:
            safe_mask = None
            all_masked = torch.zeros(B, dtype=torch.bool, device=ego_q.device)
        
        out, attn_weights = self.mha(ego_q, ctx_kv, ctx_kv, key_padding_mask=safe_mask, average_attn_weights=False)
        out = out.squeeze(1)  # (B, E)
        
        # EXTENSIVE DEBUG: Log attention weights
        if not hasattr(self, '_attn_debug_counter'):
            self._attn_debug_counter = 0
        self._attn_debug_counter += 1
        if self._attn_debug_counter <= 10:
            print(f"\n[CROSS-ATTENTION] Call {self._attn_debug_counter}:")
            print(f"  ego_q: mean={ego_q.mean().item():.4f}, std={ego_q.std().item():.4f}")
            print(f"  ctx_kv: mean={ctx_kv.mean().item():.4f}, std={ctx_kv.std().item():.4f}")
            if key_padding_mask is not None:
                valid_ctx = (~key_padding_mask).sum().item()
                print(f"  Valid context: {valid_ctx} / {key_padding_mask.numel()}")
                # attn_weights: (B, num_heads, 1, L)
                # Average across heads for display
                avg_attn = attn_weights.mean(dim=1).squeeze(1)  # (B, L)
                print(f"  Attention weights (avg over heads): mean={avg_attn.mean().item():.6f}, std={avg_attn.std().item():.6f}")
                print(f"  Attention range: min={avg_attn.min().item():.6f}, max={avg_attn.max().item():.6f}")
                # Show first sample's attention distribution
                first_attn = avg_attn[0]
                first_mask = key_padding_mask[0]
                valid_attn = first_attn[~first_mask]
                if len(valid_attn) > 0:
                    print(f"  First sample valid attention: {valid_attn[:5].tolist()}")
                    print(f"  Attention entropy: {-(valid_attn * torch.log(valid_attn + 1e-9)).sum().item():.4f}")
            print(f"  Output: mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        
        # Zero out results for samples that had all positions masked
        if all_masked.any():
            out = out * (~all_masked).float().unsqueeze(-1)
        
        return out
