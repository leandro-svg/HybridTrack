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

    def step_prior(self, sequence: Optional[torch.Tensor] = None) -> None:
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
        new_state_3d_bb_vector_lwh = self.bb3d_layers['wlh'](fused_feats)
        new_state_3d_bb_vector_y = self.bb3d_layers['x'](fused_feats)
        new_state_3d_bb_vector_x = self.bb3d_layers['y'](fused_feats)
        new_state_3d_bb_vector_z = self.bb3d_layers['z'](fused_feats)
        ry_temp = self.ry_layer(torch.cat((self.ry_layers['current'](temp_m1x_posterior_ry),
                                           self.ry_layers['previous'](temp_m1x_posterior_ry_previous),
                                           self.ry_layers['residual'](temp_m1x_posterior_ry_residual)), dim=1))
        residual_m1xprior = torch.cat((new_state_3d_bb_vector_x, new_state_3d_bb_vector_y, new_state_3d_bb_vector_z, new_state_3d_bb_vector_lwh, ry_temp), dim=1)
        residual_m1xprior = residual_m1xprior.unsqueeze(-1)
        self.m1x_prior = self.m1x_posterior + residual_m1xprior
        self.m1y = self.m1x_prior

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
        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)
        KG, Pcov = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = KG.view(self.batch_size, self.m, self.n)
        self.Pcov = Pcov

    def LKF_step(self, y: torch.Tensor, sequence: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.step_prior(sequence)
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

    def forward(self, y: torch.Tensor, sequence: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.to(self.device)
        return self.LKF_step(y, sequence)

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

    def forward(self, data: torch.Tensor, sequence: Optional[torch.Tensor] = None):
        prediction, state_prior = self.LKF_model(data, sequence)
        return prediction, state_prior, self.LKF_model.Pcov



