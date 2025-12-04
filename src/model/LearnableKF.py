"""
Learnablz Kalman Filter Model
----------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from typing import Optional, Tuple
from .context_modules import DetectionContextEncoder, HistoryContextEncoder, CrossAttentionFusion

def weights_init_xavier(m: nn.Module) -> None:
    """Apply Xavier initialization to Linear, Conv, and BatchNorm layers."""
    classname = m.__class__.__name__
    if 'Linear' in classname or 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif 'BatchNorm' in classname and getattr(m, 'affine', False):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

FEATURE_DIM = 32 

class SharedFeatureExtractor(nn.Module):
    """A single extractor with context tokens for different roles."""
    def __init__(self, input_dim, feature_dim=FEATURE_DIM, depth=4, width=128, dropout=0.15, use_gelu=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act = nn.GELU if use_gelu else nn.ReLU
        for i in range(depth):
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.LayerNorm(width))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            prev_dim = width
        layers.append(nn.Linear(width, feature_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        return self.net(x)

class LightweightSelfAttention(nn.Module):
    def __init__(self, dim, heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out

class LKF(nn.Module):
    """
    Neural Network for sequence-based Kalman filtering.
    """
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, cfg) -> None:
        self.device = torch.device('cuda' if cfg.TRAINER.USE_CUDA else 'cpu')
        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, cfg)

    def InitSystemDynamics(self, f, h, m, n) -> None:
        self.f, self.h, self.m, self.n = f, h, m, n

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, cfg) -> None:
        """Initialize Kalman Gain Network and all submodules."""
        self.seq_len_input = 1
        self.batch_size = cfg.TRAINER.BATCH_SIZE
        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)
        m, n = self.m, self.n
        in_mult = cfg.TRAINER.IN_MULT_LKF
        out_mult = cfg.TRAINER.OUT_MULT_LKF
        # GRU/FC layers for Q
        self.d_input_Q = m * in_mult
        self.d_hidden_Q = m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)
        self.FC_Q = nn.Sequential(
            nn.Linear(self.d_input_Q, self.d_hidden_Q), nn.ReLU(),
            nn.Linear(self.d_hidden_Q, self.d_hidden_Q), nn.ReLU()
        ).to(self.device)
        # GRU/FC layers for Sigma
        self.d_input_Sigma = self.d_hidden_Q + m * in_mult
        self.d_hidden_Sigma = m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
        self.FC_Sigma = nn.Sequential(
            nn.Linear(self.d_input_Sigma, self.d_hidden_Sigma), nn.ReLU(),
            nn.Linear(self.d_hidden_Sigma, self.d_hidden_Sigma), nn.ReLU()
        ).to(self.device)
        # GRU/FC layers for S
        self.d_input_S = n ** 2 + 2 * n * in_mult
        self.d_hidden_S = n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        self.FC_S = nn.Sequential(
            nn.Linear(self.d_input_S, self.d_hidden_S), nn.ReLU(),
            nn.Linear(self.d_hidden_S, self.d_hidden_S), nn.ReLU()
        ).to(self.device)
        # Fully connected layers for Kalman gain computation
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma, n ** 2), nn.ReLU()
        ).to(self.device)
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = n * m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2), nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2), nn.ReLU()
        ).to(self.device)
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_output_FC2, m ** 2), nn.ReLU()
        ).to(self.device)
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma + m ** 2, self.d_hidden_Sigma), nn.ReLU()
        ).to(self.device)
        self.FC5 = nn.Sequential(
            nn.Linear(m, m * in_mult), nn.ReLU()
        ).to(self.device)
        self.FC6 = nn.Sequential(
            nn.Linear(m, m * in_mult), nn.ReLU()
        ).to(self.device)
        self.FC7 = nn.Sequential(
            nn.Linear(2 * n, 2 * n * in_mult), nn.ReLU()
        ).to(self.device)
        # Shared feature extractor
        context_dim = 8 
        self.shared_feature_extr = SharedFeatureExtractor(m + context_dim, feature_dim=FEATURE_DIM, depth=4, width=128, dropout=0.15, use_gelu=True).to(self.device)
        self.fusion_attention = LightweightSelfAttention(FEATURE_DIM, heads=2, dropout=0.1).to(self.device)

        # Context encoders and cross-attention (optional)
        def ctx_param(name: str, default):
            if hasattr(cfg, 'CONTEXT') and cfg.CONTEXT is not None:
                return getattr(cfg.CONTEXT, name, default)
            return default
        self.use_context = bool(ctx_param('USE_CONTEXT', False))
        self.use_history = bool(ctx_param('USE_HISTORY', False))
        ctx_embed = int(ctx_param('DET_FEAT_DIM', 64))
        hist_embed = int(ctx_param('HIST_FEAT_DIM', 64))
        nheads = int(ctx_param('ATTN_HEADS', 4))
        if self.use_context:
            self.det_enc = DetectionContextEncoder(in_dim=8, embed_dim=ctx_embed).to(self.device)
            self.cross_attn_det = CrossAttentionFusion(embed_dim=ctx_embed, nhead=nheads, dropout=0.1).to(self.device)
            self.q_proj_det = nn.Linear(FEATURE_DIM, ctx_embed).to(self.device)
            # Learned null embedding to handle missing context
            self.null_ctx_token = nn.Parameter(torch.randn(1, FEATURE_DIM, device=self.device) * 0.02)
            # CRITICAL FIX: Make alpha learnable so model can adapt to context quality
            self.ctx_gate = nn.Parameter(torch.tensor(0.5, device=self.device))  # Start conservative
            # Context dropout for robustness
            self.ctx_dropout_prob = 0.3 if self.training else 0.0
        if self.use_context and self.use_history:
            self.hist_enc = HistoryContextEncoder(state_dim=m, embed_dim=hist_embed, use_transformer=True, nhead=nheads).to(self.device)
            self.cross_attn_hist = CrossAttentionFusion(embed_dim=hist_embed, nhead=nheads, dropout=0.1).to(self.device)
            self.q_proj_hist = nn.Linear(FEATURE_DIM, hist_embed).to(self.device)
        # Project attended context to FEATURE_DIM to fuse with ego feats
        self.ctx_proj = nn.Linear((ctx_embed if self.use_context else 0) + (hist_embed if (self.use_context and self.use_history) else 0), FEATURE_DIM).to(self.device) if self.use_context else None
        self.uncertainty_scale = nn.Parameter(torch.ones(1, requires_grad=True, device=self.device))
        self.ry_layers = nn.ModuleDict({
            'current': nn.Sequential(
                nn.Linear(1, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, FEATURE_DIM), nn.ReLU()
            ).to(self.device),
            'previous': nn.Sequential(
                nn.Linear(1, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, FEATURE_DIM), nn.ReLU()
            ).to(self.device),
            'residual': nn.Sequential(
                nn.Linear(1, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, FEATURE_DIM), nn.ReLU()
            ).to(self.device)
        })
        self.ry_layer = nn.Sequential(
            nn.Linear(3 * FEATURE_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Identity()
        ).to(self.device)
        # 3D bounding box layers
        self.bb3d_layers = nn.ModuleDict({
            'wlh': nn.Sequential(
                nn.Linear(FEATURE_DIM, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 3), nn.Identity()
            ).to(self.device),
            'x': nn.Sequential(
                nn.Linear(FEATURE_DIM, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Identity()
            ).to(self.device),
            'y': nn.Sequential(
                nn.Linear(FEATURE_DIM, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Identity()
            ).to(self.device),
            'z': nn.Sequential(
                nn.Linear(FEATURE_DIM, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Identity()
            ).to(self.device)
        })
        self.p_cov_linear = nn.Linear(16, 1)
        self.apply(weights_init_xavier)
        for layer in [self.FC1, self.FC2, self.FC3, self.FC4, self.FC5, self.FC6, self.FC7,
                      self.shared_feature_extr, self.fusion_attention,
                      *self.ry_layers.values(), self.ry_layer, *self.bb3d_layers.values(), self.FC_S, self.FC_Q, self.FC_Sigma]:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def update_batch_size(self, new_bs: int):
        """
        Update internal batch size and expand GRU hidden states / state tensors if needed.
        Call this when tracker uses a larger pool than training batch size.
        """
        if new_bs == self.batch_size:
            return
        self.batch_size = new_bs
        def expand(t):
            # t: (1, 1, D) or (1, old_bs, D)
            if t.size(1) == new_bs:
                return t
            base = t[:, 0:1, :].clone()
            return base.repeat(1, new_bs, 1)
        self.h_S = expand(self.h_S)
        self.h_Sigma = expand(self.h_Sigma)
        self.h_Q = expand(self.h_Q)
        # Expand state tensors
        def expand_state(s):
            if s.size(0) == new_bs:
                return s
            base = s[0:1, ...].clone()
            return base.repeat(new_bs, 1, 1)
        for name in ['m1x_posterior', 'm1x_posterior_previous', 'm1x_posterior_previous_previous',
                     'm1x_prior_previous', 'm1y', 'm1x_prior', 'y_previous']:
            if hasattr(self, name):
                s = getattr(self, name)
                if s.dim() == 3:  # (B, dim, 1)
                    setattr(self, name, expand_state(s))

    def normalize_angles(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize angles to [-1, 1] range."""
        return data / np.pi

    def denormalize_angles(self, predicted_angles: torch.Tensor) -> torch.Tensor:
        return predicted_angles * np.pi

    def InitSequence(self, M1_0: torch.Tensor, T: int) -> None:
        """Initialize sequence state tensors."""
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior - 1e-6
        self.m1x_posterior_previous_previous = self.m1x_posterior - 2e-6
        self.m1x_prior_previous = self.m1x_posterior - 1e-6
        self.y_previous = self.h(self.m1x_posterior)
        self.fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)
        self.fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)

    def step_prior(self,
                   sequence: Optional[torch.Tensor] = None,
                   det_context: Optional[torch.Tensor] = None,
                   det_mask: Optional[torch.Tensor] = None,
                   hist_context: Optional[torch.Tensor] = None,
                   hist_mask: Optional[torch.Tensor] = None,
                   increment_step: bool = True,
                   return_context: bool = False) -> Optional[torch.Tensor]:
        """Predict the prior state for the next time step."""
        temp_m1x_posterior_ry = self.m1x_posterior[:, 6]
        temp_m1x_posterior_ry_previous = self.m1x_posterior_previous[:, 6]
        temp_m1x_posterior_ry_residual = self.m1x_posterior[:, 6] - self.m1x_posterior_previous[:, 6]
        # Use shared feature extractor with context tokens
        x = self.m1x_posterior.squeeze(-1)
        x_prev = self.m1x_posterior_previous.squeeze(-1)
        x_prev_prev = self.m1x_posterior_previous_previous.squeeze(-1)
        x_prior = self.m1x_prior_previous.squeeze(-1)
        residual = (self.m1x_posterior - self.m1x_posterior_previous).squeeze(-1)
        residual_prev = (self.m1x_posterior_previous - self.m1x_posterior_previous_previous).squeeze(-1)
        residual_prior_post = (self.m1x_posterior - self.m1x_prior_previous).squeeze(-1)
        second_order_diff = (x - 2 * x_prev + x_prev_prev)
        ctxs = [torch.eye(8, device=x.device)[i].unsqueeze(0).repeat(x.size(0), 1) for i in range(8)]
        feats = [
            self.shared_feature_extr(x, ctxs[0]),
            self.shared_feature_extr(x_prev, ctxs[1]),
            self.shared_feature_extr(x_prev_prev, ctxs[2]),
            self.shared_feature_extr(second_order_diff, ctxs[3]),
            self.shared_feature_extr(x_prior, ctxs[4]),
            self.shared_feature_extr(residual, ctxs[5]),
            self.shared_feature_extr(residual_prev, ctxs[6]),
            self.shared_feature_extr(residual_prior_post, ctxs[7]),
        ]
        feats_stack = torch.stack(feats, dim=1)  # (batch, seq=8, feat_dim)
        fused_feats = self.fusion_attention(feats_stack).mean(dim=1)

        # Optional cross-attention with detection and history contexts
        if self.use_context:
            B = fused_feats.size(0)
            ego_q_det = fused_feats.unsqueeze(1)  # (B, 1, FEATURE_DIM)
            ctx_feats_list = []
            masks_list = []
            # Detection context
            if det_context is not None and det_context.numel() > 0:
                # det_context: (B, N, 8), det_mask: (B, N) True for pad
                det_context = det_context.to(self.device)
                
                # Check if context has any real (non-zero) detections
                # If all detections are zeros (padded), skip processing
                has_real_dets = det_context.abs().sum(dim=-1).max() > 1e-6
                
                if has_real_dets:
                    det_kv = self.det_enc(det_context)  # (B, N, Cdet)
                    # Project ego_q to det space if needed
                    ego_det_q = self.q_proj_det(fused_feats).unsqueeze(1)
                    # Use provided mask (True=pad)
                    det_kpm = det_mask.to(self.device) if det_mask is not None else None
                    det_attn = self.cross_attn_det(ego_det_q, det_kv, key_padding_mask=det_kpm)  # (B, Cdet)
                    ctx_feats_list.append(det_attn)
                    masks_list.append(det_kpm)
            # History context
            if self.use_history and hist_context is not None and hist_context.numel() > 0:
                # hist_context: (B, K, H, 7), hist_mask: (B, K) True=pad
                hist_context = hist_context.to(self.device)
                hist_kv = self.hist_enc(hist_context, hist_mask)  # (B, K, Chist)
                ego_hist_q = self.q_proj_hist(fused_feats).unsqueeze(1)
                hist_kpm = hist_mask.to(self.device) if hist_mask is not None else None
                hist_attn = self.cross_attn_hist(ego_hist_q, hist_kv, key_padding_mask=hist_kpm)  # (B, Chist)
                ctx_feats_list.append(hist_attn)

            if ctx_feats_list:
                ctx_concat = torch.cat(ctx_feats_list, dim=-1)
                # Safety: clean any NaN/Inf from attention outputs
                ctx_concat = torch.nan_to_num(ctx_concat, nan=0.0, posinf=0.0, neginf=0.0)
                ctx_proj = self.ctx_proj(ctx_concat)  # (B, FEATURE_DIM)
                # Safety: clean projection output
                ctx_proj = torch.nan_to_num(ctx_proj, nan=0.0, posinf=0.0, neginf=0.0)
                
                # CRITICAL FIX: Apply dropout during training for robustness
                if self.training and hasattr(self, 'ctx_dropout_prob'):
                    dropout_mask = torch.rand(B, 1, device=self.device) > self.ctx_dropout_prob
                    ctx_proj = ctx_proj * dropout_mask.float()
                
                # EXTENSIVE DEBUG: Log context processing
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 1
                if self._debug_counter <= 10:
                    print("\n" + "="*80)
                    print(f"[MODEL CTX FUSION] Step {self._debug_counter}")
                    print("="*80)
                    print("  INPUTS:")
                    print(f"    det_context: shape={det_context.shape if det_context is not None else 'None'}")
                    if det_context is not None:
                        print(f"    det_context stats: mean={det_context.mean().item():.4f}, std={det_context.std().item():.4f}")
                        print(f"    det_context range: min={det_context.min().item():.4f}, max={det_context.max().item():.4f}")
                        print(f"    det_context xyz: mean={det_context[:,:,:3].mean().item():.4f}, std={det_context[:,:,:3].std().item():.4f}")
                        print(f"    det_context scores: mean={det_context[:,:,7].mean().item():.4f}, std={det_context[:,:,7].std().item():.4f}")
                        print(f"    det_mask: {det_mask.sum().item() if det_mask is not None else 'None'} padded out of {det_mask.numel() if det_mask is not None else 0}")
                        valid_ctx = (~det_mask).sum().item() if det_mask is not None else 0
                        print(f"    Valid context entries: {valid_ctx}")
                    
                    print("\n  ENCODING:")
                    if 'det_kv' in locals():
                        print(f"    det_kv (encoded): mean={det_kv.mean().item():.4f}, std={det_kv.std().item():.4f}")
                        print(f"    det_kv range: min={det_kv.min().item():.4f}, max={det_kv.max().item():.4f}")
                    if 'ego_det_q' in locals():
                        print(f"    ego_det_q (query): mean={ego_det_q.mean().item():.4f}, std={ego_det_q.std().item():.4f}")
                    if 'det_attn' in locals():
                        print(f"    det_attn (after attn): mean={det_attn.mean().item():.4f}, std={det_attn.std().item():.4f}")
                    
                    print("\n  FUSION:")
                    print(f"    ctx_concat: mean={ctx_concat.mean().item():.4f}, std={ctx_concat.std().item():.4f}")
                    print(f"    ctx_proj: mean={ctx_proj.mean().item():.4f}, std={ctx_proj.std().item():.4f}")
                    print(f"    ctx_proj range: min={ctx_proj.min().item():.4f}, max={ctx_proj.max().item():.4f}")
                    gate_raw = self.ctx_gate.item()
                    gate_value = torch.sigmoid(self.ctx_gate).item()
                    print(f"    ctx_gate (raw): {gate_raw:.6f}, sigmoid: {gate_value:.6f}")
                    ctx_contribution = (gate_value * ctx_proj).abs().mean().item()
                    print(f"    Context contribution magnitude: {ctx_contribution:.6f}")
                    print(f"    fused_feats (before ctx): mean={fused_feats.mean().item():.4f}, std={fused_feats.std().item():.4f}")
                    print("="*80 + "\n")
            else:
                # Use learned null token broadcast when no context available
                ctx_proj = self.null_ctx_token.repeat(fused_feats.size(0), 1)
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 1
                if self._debug_counter <= 10:
                    print(f"\n[MODEL CTX FUSION] Step {self._debug_counter}: NO CONTEXT - Using null token")
            
            # CRITICAL FIX: Use learnable gate (sigmoid for 0-1 range)
            fused_feats_before = fused_feats.clone() if self._debug_counter <= 10 else None
            gate_value = torch.sigmoid(self.ctx_gate)  # Learnable, bounded to [0,1]
            gate_term = gate_value * ctx_proj
            fused_feats = fused_feats + gate_term
            
            # EXTENSIVE DEBUG: Log the actual fusion impact
            if self._debug_counter <= 10:
                print("  AFTER FUSION:")
                print(f"    Learnable gate value: {gate_value.item():.6f}")
                print(f"    gate_term: mean={gate_term.mean().item():.6f}, std={gate_term.std().item():.6f}")
                print(f"    gate_term range: min={gate_term.min().item():.6f}, max={gate_term.max().item():.6f}")
                print(f"    fused_feats (after ctx): mean={fused_feats.mean().item():.4f}, std={fused_feats.std().item():.4f}")
                if fused_feats_before is not None:
                    diff = (fused_feats - fused_feats_before).abs().mean().item()
                    print(f"    Absolute change in fused_feats: {diff:.6f}")
                    rel_change = diff / (fused_feats_before.abs().mean().item() + 1e-8)
                    print(f"    Relative change: {rel_change:.6f}")
                print("="*80 + "\n")
        new_state_3d_bb_vector_lwh = self.bb3d_layers['wlh'](fused_feats)
        new_state_3d_bb_vector_y = self.bb3d_layers['x'](fused_feats)
        new_state_3d_bb_vector_x = self.bb3d_layers['y'](fused_feats)
        new_state_3d_bb_vector_z = self.bb3d_layers['z'](fused_feats)
        ry_temp = self.ry_layer(torch.cat((self.ry_layers['current'](temp_m1x_posterior_ry),
                                           self.ry_layers['previous'](temp_m1x_posterior_ry_previous),
                                           self.ry_layers['residual'](temp_m1x_posterior_ry_residual)), dim=1))
        residual_m1xprior = torch.cat((new_state_3d_bb_vector_x, new_state_3d_bb_vector_y, new_state_3d_bb_vector_z, new_state_3d_bb_vector_lwh, ry_temp), dim=1)
        residual_m1xprior = residual_m1xprior.unsqueeze(-1)
        if increment_step:
            self.m1x_prior = self.m1x_posterior + residual_m1xprior
            self.m1y = self.m1x_prior
        if return_context:
            return fused_feats

    def _safe_normalize(self, x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
        """
        Safe L2 normalization that handles zero-norm and NaN cases.
        Returns zeros for vectors with near-zero norm instead of NaN.
        """
        # Replace any NaN/Inf in input
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Compute norm
        norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
        # Mask for near-zero norms
        mask = norm > eps
        # Safe division: only normalize where norm is large enough
        result = torch.where(mask, x / (norm + eps), torch.zeros_like(x))
        return result

    def step_KGain_est(self, y: torch.Tensor) -> None:
        """Estimate Kalman gain given the current observation."""
        y = y.to(self.device)
        self.y_previous = self.y_previous.to(self.device)
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        if self.y_previous.dim() == 2:
            self.y_previous = self.y_previous.unsqueeze(-1)
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y.to(self.device), 2)
        fw_evol_diff = torch.squeeze(self.m1x_posterior.to(self.device), 2) - torch.squeeze(self.m1x_posterior_previous.to(self.device), 2)
        fw_update_diff = torch.squeeze(self.m1x_posterior.to(self.device), 2) - torch.squeeze(self.m1x_prior_previous.to(self.device), 2)
        # Safe normalization: handle zero-norm and NaN cases to prevent backward pass NaN
        obs_diff = self._safe_normalize(obs_diff)
        obs_innov_diff = self._safe_normalize(obs_innov_diff)
        fw_evol_diff = self._safe_normalize(fw_evol_diff)
        fw_update_diff = self._safe_normalize(fw_update_diff)
        KG, Pcov = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = KG.view(self.batch_size, self.m, self.n)
        self.Pcov = Pcov

    def LKF_step(self, y: torch.Tensor, sequence: Optional[torch.Tensor],
                 det_context: Optional[torch.Tensor] = None,
                 det_mask: Optional[torch.Tensor] = None,
                 hist_context: Optional[torch.Tensor] = None,
                 hist_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.step_prior(sequence, det_context=det_context, det_mask=det_mask, hist_context=hist_context, hist_mask=hist_mask)
        self.step_KGain_est(y)
        self.m1x_prior_previous = self.m1x_prior
        dy = y.unsqueeze(2) - self.m1y
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous_previous = self.m1x_posterior_previous
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV
        self.y_previous = y.unsqueeze(-1)
        return self.m1x_posterior, self.m1x_prior

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1], device=self.device)
            expanded[0, :, :] = x
            return expanded
        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)
        out_FC6 = self.FC6(fw_update_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)
        out_FC1 = self.FC1(out_Sigma)
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2) * self.uncertainty_scale
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)
        self.h_Sigma = out_FC4
        return out_FC2, out_FC4

    def forward(self, y: torch.Tensor, sequence: Optional[torch.Tensor],
                det_context: Optional[torch.Tensor] = None,
                det_mask: Optional[torch.Tensor] = None,
                hist_context: Optional[torch.Tensor] = None,
                hist_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.to(self.device)
        return self.LKF_step(y, sequence, det_context=det_context, det_mask=det_mask, hist_context=hist_context, hist_mask=hist_mask)

    def init_hidden_LKF(self) -> None:
        """Initialize hidden states for all GRUs."""
        m, n = self.m, self.n
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        M1_0 = torch.ones(m, 1, device=self.device)
        self.h_posterior = M1_0.squeeze(-1).flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)
        h_state_0 = torch.zeros((512, 1), device=self.device)
        self.h_state = h_state_0.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)

class LEARNABLEKF(nn.Module):
    """
    Wrapper for LKF, providing a high-level interface for pose estimation.
    """
    def __init__(self, SysModel, cfg):
        super().__init__()
        self.device = torch.device('cuda' if cfg.TRAINER.USE_CUDA else 'cpu')
        self.LKF_model = LKF()
        self.LKF_model.NNBuild(SysModel, cfg)

    def init_hidden_LKF(self) -> None:
        self.LKF_model.init_hidden_LKF()

    def forward(self, data: torch.Tensor, sequence: Optional[torch.Tensor] = None,
                det_context: Optional[torch.Tensor] = None,
                det_mask: Optional[torch.Tensor] = None,
                hist_context: Optional[torch.Tensor] = None,
                hist_mask: Optional[torch.Tensor] = None):
        prediction, state_prior = self.LKF_model(data, sequence, det_context=det_context, det_mask=det_mask, hist_context=hist_context, hist_mask=hist_mask)
        return prediction, state_prior, self.LKF_model.Pcov



