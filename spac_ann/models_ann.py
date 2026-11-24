"""PyTorch deep-learning inversion architectures for 1D Vs(z) and engineering bedrock Z760:
  - MDNInversionNet: Probabilistic K-component Gaussian Mixture Density Network
  - JointNetDispFormer: Masked dual-branch 1D-ResNet + Transformer for Dispersion + HVSR
  - ConditionalNormalizingFlow3C: 8-layer orthogonal-mixed RealNVP Neural Posterior Estimator
    for joint Rayleigh + Love + RayDec/DFA HVSR correlated posteriors
  - synthesize_domain_aligned_hvsr: Field-realistic HVSR/ellipticity augmentation
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_C_MEAN = math.log(450.0)
LOG_C_STD = 0.65
LOG_VS_MEAN = math.log(550.0)
LOG_VS_STD = 0.75
LOG_F0_MEAN = math.log(1.0)
LOG_F0_STD = 0.90
LOG_Z_MIN = math.log(8.0)
LOG_Z_MAX = math.log(750.0)


def norm_log_z(z_m: np.ndarray | torch.Tensor):
    if isinstance(z_m, torch.Tensor):
        return (torch.log(torch.clamp(z_m, 8.0, 750.0)) - LOG_Z_MIN) / (LOG_Z_MAX - LOG_Z_MIN)
    return (np.log(np.clip(z_m, 8.0, 750.0)) - LOG_Z_MIN) / (LOG_Z_MAX - LOG_Z_MIN)


def denorm_log_z(z_norm: np.ndarray | torch.Tensor):
    if isinstance(z_norm, torch.Tensor):
        return torch.exp(z_norm * (LOG_Z_MAX - LOG_Z_MIN) + LOG_Z_MIN)
    return np.exp(z_norm * (LOG_Z_MAX - LOG_Z_MIN) + LOG_Z_MIN)


def synthesize_domain_aligned_hvsr(
    f_grid: np.ndarray,
    f0_sh: np.ndarray,
    a0_sh: np.ndarray,
    ellip: np.ndarray,
    rng: np.random.Generator,
    augment: bool = True,
    force_disp_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize field-realistic 3C HVSR curves (diffuse-field shelf + peak + high-f decay)
    lightly blended with trough-clamped Rayleigh ellipticity to prevent synthetic-to-field domain shift.
    Returns (hv_log, f0_bump, hv_valid, hv_curve_valid) + perturbed (f0_obs, a0_obs) inside dict?
    Returns (hv_seq_3ch, f0_obs, a0_obs, hv_valid).
    """
    B = len(f0_sh)
    n_f = len(f_grid)
    log_f_row = np.log(f_grid)[None, :].astype(np.float32)

    if force_disp_only:
        hv_valid = np.zeros(B, dtype=np.float32)
        hv_curve_valid = np.zeros(B, dtype=np.float32)
        f0_obs = np.clip(f0_sh, 0.20, 14.0).astype(np.float32)
        a0_obs = np.clip(a0_sh, 1.0, 12.0).astype(np.float32)
    elif augment:
        u_hv = rng.random(B)
        hv_valid = (u_hv >= 0.25).astype(np.float32)
        hv_curve_valid = (u_hv >= 0.55).astype(np.float32)
        f0_obs = np.clip(f0_sh * np.exp(rng.normal(0.0, 0.08, size=B)), 0.20, 14.0).astype(np.float32)
        a0_obs = np.clip(a0_sh * np.exp(rng.normal(0.0, 0.28, size=B)), 1.5, 12.0).astype(np.float32)
    else:
        hv_valid = np.ones(B, dtype=np.float32)
        hv_curve_valid = np.ones(B, dtype=np.float32)
        f0_obs = np.clip(f0_sh * np.exp(rng.normal(0.0, 0.05, size=B)), 0.20, 14.0).astype(np.float32)
        a0_obs = np.clip(a0_sh * np.exp(rng.normal(0.0, 0.15, size=B)), 1.5, 12.0).astype(np.float32)

    df = log_f_row - np.log(f0_obs)[:, None]
    base_lo = rng.uniform(0.9, 2.6, size=(B, 1)).astype(np.float32)
    base_hi = rng.uniform(0.28, 1.15, size=(B, 1)).astype(np.float32)
    trans = 1.0 / (1.0 + np.exp(-3.0 * (df - 0.35)))
    baseline = base_lo * (1.0 - trans) + base_hi * trans
    peak_w = rng.uniform(0.18, 0.48, size=(B, 1)).astype(np.float32)
    peak_amp = np.maximum(0.5, a0_obs[:, None] - baseline)
    hv_syn = baseline + peak_amp * np.exp(-0.5 * (df / peak_w) ** 2)

    if augment:
        has_f1 = (rng.random(B) < 0.25)[:, None]
        f1_shift = rng.uniform(0.40, 1.15, size=(B, 1)).astype(np.float32)
        a1_amp = rng.uniform(0.4, 2.2, size=(B, 1)).astype(np.float32)
        hv_syn = hv_syn + np.where(has_f1, a1_amp * np.exp(-0.5 * ((df - f1_shift) / 0.28) ** 2), 0.0)

    el_safe = np.where(np.isfinite(ellip) & (ellip > 1e-3), np.clip(ellip, 0.35, 10.0), hv_syn)
    beta_el = rng.uniform(0.0, 0.30, size=(B, 1)).astype(np.float32) if augment else 0.15
    hv_blend = (1.0 - beta_el) * hv_syn + beta_el * el_safe
    hv_log = np.log10(np.clip(hv_blend, 0.20, 15.0)).astype(np.float32)
    if augment:
        hv_log = hv_log + rng.normal(0.0, 0.06, size=(B, n_f)).astype(np.float32)
    hv_log = np.clip(hv_log, -0.75, 1.25) * hv_curve_valid[:, None]

    f0_bump = np.exp(-0.5 * (df / 0.28) ** 2).astype(np.float32) * hv_valid[:, None]
    hv_mask_vec = np.broadcast_to(hv_valid[:, None], (B, n_f)).copy()
    hv_seq = np.stack([hv_log, f0_bump, hv_mask_vec], axis=1).astype(np.float32)
    return hv_seq, f0_obs, a0_obs, hv_valid


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MDNInversionNet(nn.Module):
    """Probabilistic Mixture Density Network with K Gaussian components for posterior
    Vs(z), Z760_first, Z760_perm, Z1000, and bedrock existence probabilities.
    """

    def __init__(self, n_f: int = 64, n_z: int = 48, n_components: int = 6, hidden: int = 384):
        super().__init__()
        self.n_z = n_z
        self.n_out_cont = n_z + 3
        self.K = n_components

        self.disp_conv = nn.Sequential(
            nn.Conv1d(4, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.hv_conv = nn.Sequential(
            nn.Conv1d(3, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 32, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )
        in_dim = 64 * 16 + 32 * 16 + 6
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(
            ResMLPBlock(hidden, dropout=0.08),
            ResMLPBlock(hidden, dropout=0.08),
            ResMLPBlock(hidden, dropout=0.08),
            ResMLPBlock(hidden, dropout=0.05),
        )
        self.norm = nn.LayerNorm(hidden)
        self.pi_head = nn.Linear(hidden, self.K)
        self.mu_head = nn.Linear(hidden, self.K * self.n_out_cont)
        self.log_sigma_head = nn.Linear(hidden, self.K * self.n_out_cont)
        self.cls_head = nn.Linear(hidden, 2)

    def forward(self, disp_seq: torch.Tensor, hv_seq: torch.Tensor, hv_scalars: torch.Tensor):
        fd = self.disp_conv(disp_seq).flatten(1)
        fh = self.hv_conv(hv_seq).flatten(1)
        h = self.norm(self.blocks(self.in_proj(torch.cat([fd, fh, hv_scalars], dim=1))))
        B = h.size(0)
        log_pi = F.log_softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(B, self.K, self.n_out_cont)
        sigma = F.softplus(self.log_sigma_head(h)).view(B, self.K, self.n_out_cont) + 0.025
        cls_logits = self.cls_head(h)
        return log_pi, mu, sigma, cls_logits

    @torch.no_grad()
    def sample_posterior(
        self, disp_seq: torch.Tensor, hv_seq: torch.Tensor, hv_scalars: torch.Tensor, n_samples: int = 4000
    ) -> Dict[str, np.ndarray]:
        self.eval()
        log_pi, mu, sigma, cls_logits = self.forward(disp_seq, hv_seq, hv_scalars)
        probs = torch.exp(log_pi)
        comp_idx = torch.multinomial(probs, n_samples, replacement=True)
        idx_exp = comp_idx.unsqueeze(-1).expand(-1, -1, self.n_out_cont)
        samples = torch.gather(mu, 1, idx_exp) + torch.gather(sigma, 1, idx_exp) * torch.randn_like(
            torch.gather(mu, 1, idx_exp)
        )
        cls_p = torch.sigmoid(cls_logits)
        return {
            "vs_samples": np.exp(samples[:, :, : self.n_z].cpu().numpy() * LOG_VS_STD + LOG_VS_MEAN),
            "z760_first_samples": denorm_log_z(samples[:, :, self.n_z].cpu().numpy()),
            "z760_perm_samples": denorm_log_z(samples[:, :, self.n_z + 1].cpu().numpy()),
            "z1000_perm_samples": denorm_log_z(samples[:, :, self.n_z + 2].cpu().numpy()),
            "p_has_z760": cls_p[:, 0].cpu().numpy(),
            "p_has_z1000": cls_p[:, 1].cpu().numpy(),
        }


def mdn_loss(log_pi, mu, sigma, cls_logits, target, n_z: int = 48):
    y_cont = target[:, : n_z + 3]
    y_cls = target[:, n_z + 3 :]
    has_760, has_1000 = y_cls[:, 0], y_cls[:, 1]
    w = torch.ones_like(y_cont)
    w[:, n_z] = 10.0 * has_760 + 0.5
    w[:, n_z + 1] = 14.0 * has_760 + 0.5
    w[:, n_z + 2] = 6.0 * has_1000 + 0.3

    log_prob_d = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * ((y_cont.unsqueeze(1) - mu) / sigma) ** 2
    log_prob_k = (log_prob_d * w.unsqueeze(1)).sum(dim=-1) / w.sum(dim=-1, keepdim=True)
    nll = -torch.logsumexp(log_pi + log_prob_k, dim=-1).mean()
    return nll + 0.35 * F.binary_cross_entropy_with_logits(cls_logits, y_cls)


class ResConv1DBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.05):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class JointNetDispFormer(nn.Module):
    """Multi-modal dual-branch joint inversion network: Masked 1D-ResNet + Transformer
    dispersion encoder fused with HVSR / f0 encoder.
    """

    def __init__(self, n_f: int = 64, n_z: int = 48, d_model: int = 128):
        super().__init__()
        self.n_z = n_z
        self.disp_stem = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            ResConv1DBlock(d_model, kernel_size=5, dilation=1),
            ResConv1DBlock(d_model, kernel_size=5, dilation=2),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 3,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.disp_transformer = nn.TransformerEncoder(enc_layer, num_layers=3, enable_nested_tensor=False)
        self.disp_pool_q = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.disp_attn_pool = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        self.hv_stem = nn.Sequential(
            nn.Conv1d(3, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.GELU(),
            ResConv1DBlock(48, kernel_size=5, dilation=1),
            nn.AdaptiveAvgPool1d(4),
        )
        self.hv_scalar_mlp = nn.Sequential(nn.Linear(6, 48), nn.GELU(), nn.Linear(48, 64), nn.GELU())
        fused_dim = 2 * d_model + 48 * 4 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            ResMLPBlock(384, dropout=0.08),
            ResMLPBlock(384, dropout=0.06),
            nn.LayerNorm(384),
        )
        self.vs_head = nn.Linear(384, n_z)
        self.z_head = nn.Linear(384, 3)
        self.cls_head = nn.Linear(384, 2)

    def forward(self, disp_seq: torch.Tensor, hv_seq: torch.Tensor, hv_scalars: torch.Tensor):
        B = disp_seq.size(0)
        xd = self.disp_transformer(self.disp_stem(disp_seq).transpose(1, 2))
        xd_pool, _ = self.disp_attn_pool(self.disp_pool_q.expand(B, -1, -1), xd, xd)
        xh = self.hv_stem(hv_seq).flatten(1)
        xs = self.hv_scalar_mlp(hv_scalars)
        f = self.fusion(torch.cat([xd_pool.flatten(1), xh, xs], dim=1))
        return self.vs_head(f), self.z_head(f), self.cls_head(f)


def jointnet_loss(vs_pred, z_pred, cls_logits, target, n_z: int = 48):
    y_vs = target[:, :n_z]
    y_z = target[:, n_z : n_z + 3]
    y_cls = target[:, n_z + 3 :]
    has_760, has_1000 = y_cls[:, 0], y_cls[:, 1]
    l_vs = F.smooth_l1_loss(vs_pred, y_vs, beta=0.15)
    l_grad = F.smooth_l1_loss(vs_pred[:, 1:] - vs_pred[:, :-1], y_vs[:, 1:] - y_vs[:, :-1], beta=0.15)
    err_z = F.smooth_l1_loss(z_pred, y_z, reduction="none", beta=0.05)
    w_z = torch.stack([has_760 + 0.05, 1.5 * has_760 + 0.05, 0.7 * has_1000 + 0.03], dim=1)
    l_z = (err_z * w_z).sum() / w_z.sum()
    return l_vs + 0.25 * l_grad + 2.2 * l_z + 0.35 * F.binary_cross_entropy_with_logits(cls_logits, y_cls)


class ConditionalAffineCoupling(nn.Module):
    """Exact invertible Conditional RealNVP Coupling Layer with fixed orthogonal mixing Q."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 256, seed: int = 0):
        super().__init__()
        self.dim = dim
        self.d1 = dim // 2
        self.d2 = dim - self.d1
        gen = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))
        self.register_buffer("Q", q)
        self.net = nn.Sequential(
            nn.Linear(self.d1 + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.d2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_rot = y @ self.Q
        y1, y2 = y_rot[:, : self.d1], y_rot[:, self.d1 :]
        st = self.net(torch.cat([y1, cond], dim=1))
        raw_s, t = st[:, : self.d2], st[:, self.d2 :]
        s = 1.8 * torch.tanh(raw_s / 1.8)
        z = torch.cat([y1, y2 * torch.exp(s) + t], dim=1)
        return z, s.sum(dim=1)

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[:, : self.d1], z[:, self.d1 :]
        st = self.net(torch.cat([z1, cond], dim=1))
        raw_s, t = st[:, : self.d2], st[:, self.d2 :]
        s = 1.8 * torch.tanh(raw_s / 1.8)
        y_rot = torch.cat([z1, (z2 - t) * torch.exp(-s)], dim=1)
        return y_rot @ self.Q.T


class ConditionalNormalizingFlow3C(nn.Module):
    """Full-Covariance Conditional Normalizing Flow (NPE) for 3C Rayleigh + Love + RayDec/DFA HVSR."""

    def __init__(self, n_f: int = 64, n_z: int = 48, n_flows: int = 8, cond_dim: int = 256):
        super().__init__()
        self.n_z = n_z
        self.y_dim = n_z + 3
        self.disp_enc = nn.Sequential(
            nn.Conv1d(7, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.hv_enc = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )
        in_dim = 96 * 16 + 32 * 16 + 8
        self.cond_proj = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
        )
        self.flows = nn.ModuleList(
            [ConditionalAffineCoupling(self.y_dim, cond_dim, hidden=256, seed=700 + i * 31) for i in range(n_flows)]
        )
        self.cls_head = nn.Linear(cond_dim, 2)

    def encode(self, disp_seq: torch.Tensor, hv_seq: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.cond_proj(
            torch.cat([self.disp_enc(disp_seq).flatten(1), self.hv_enc(hv_seq).flatten(1), scalars], dim=1)
        )

    def log_prob(self, y: torch.Tensor, disp_seq: torch.Tensor, hv_seq: torch.Tensor, scalars: torch.Tensor):
        cond = self.encode(disp_seq, hv_seq, scalars)
        z = y
        log_det = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)
        for flow in self.flows:
            z, ld = flow(z, cond)
            log_det = log_det + ld
        log_pz = -0.5 * (self.y_dim * math.log(2.0 * math.pi) + (z * z).sum(dim=1))
        return log_pz + log_det, self.cls_head(cond)

    @torch.no_grad()
    def sample_posterior(
        self, disp_seq: torch.Tensor, hv_seq: torch.Tensor, scalars: torch.Tensor, n_samples: int = 4000
    ) -> Dict[str, np.ndarray]:
        self.eval()
        cond = self.encode(disp_seq, hv_seq, scalars)
        B, C = cond.shape
        cond_rep = cond.unsqueeze(1).expand(B, n_samples, C).reshape(B * n_samples, C)
        y = torch.randn(B * n_samples, self.y_dim, device=cond.device)
        for flow in reversed(self.flows):
            y = flow.inverse(y, cond_rep)
        y = y.view(B, n_samples, self.y_dim).cpu().numpy()
        cls_p = torch.sigmoid(self.cls_head(cond)).cpu().numpy()
        return {
            "vs_samples": np.exp(np.clip(y[:, :, : self.n_z] * LOG_VS_STD + LOG_VS_MEAN, math.log(60.0), math.log(3500.0))),
            "z760_first_samples": denorm_log_z(np.clip(y[:, :, self.n_z], -0.1, 1.15)),
            "z760_perm_samples": denorm_log_z(np.clip(y[:, :, self.n_z + 1], -0.1, 1.15)),
            "z1000_perm_samples": denorm_log_z(np.clip(y[:, :, self.n_z + 2], -0.1, 1.15)),
            "p_has_z760": cls_p[:, 0],
            "p_has_z1000": cls_p[:, 1],
        }
