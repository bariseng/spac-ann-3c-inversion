"""Mode-number-free PyTorch automatic-differentiation surface-wave dispersion
determinants (ADsurf / Maraschini et al., 2010) and two-stage joint inversion.

Provides:
  - ``brocher_vp_rho_torch``: differentiable Brocher (2005) Vp(Vs) and rho(Vp).
  - ``love_det_torch``: pole-free, sinc/tanh-regularized Love secular determinant.
  - ``rayleigh_det_plucker_torch``: overflow-free Plucker compound Rayleigh determinant.
  - ``newton_velocity_residual_torch``: mode-number-free Newton velocity distance.
  - ``sh_transfer_peak_torch``: differentiable 1D viscoelastic SH transfer function.
  - ``invert_adsurf_joint``: Stage-1 Adam autodiff + Stage-2 Powell DFA/SPAC polish.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from disba import Ellipticity, PhaseDispersion
from scipy import optimize, special

from .physics import Z_GRID_DEFAULT, brocher_vp_rho, compute_profile_metrics, sh_transfer_f0
from .raydec_dfa import compute_dfa_hvsr_full

PAIRS_6 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def brocher_vp_rho_torch(
    h_m: torch.Tensor, vs_ms: torch.Tensor, z_wt: float = 6.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Brocher (2005) Vp(Vs) and rho(Vp) with shallow water table."""
    n = vs_ms.shape[0]
    tops = torch.zeros(n, dtype=vs_ms.dtype, device=vs_ms.device)
    if n > 1:
        tops[1:] = torch.cumsum(h_m[:-1], dim=0)
    z_mids = tops.clone()
    if n > 1:
        z_mids[:-1] = tops[:-1] + 0.5 * h_m[:-1]
    z_mids[-1] = tops[-1] + 50.0

    vsk = vs_ms / 1000.0
    vp_b = (0.9409 + 2.0947 * vsk - 0.8206 * vsk**2 + 0.2683 * vsk**3 - 0.0251 * vsk**4) * 1000.0
    sat = z_mids >= z_wt
    vp_sat = torch.maximum(vp_b, torch.full_like(vp_b, 1550.0))
    vp_unsat = torch.maximum(vs_ms * 1.9, torch.minimum(vp_b, torch.full_like(vp_b, 1450.0)))
    vp = torch.maximum(torch.where(sat, vp_sat, vp_unsat), vs_ms * 1.50)

    vpk = vp / 1000.0
    rho_b = 1.6612 * vpk - 0.4721 * vpk**2 + 0.0671 * vpk**3 - 0.0043 * vpk**4 + 0.000106 * vpk**5
    rho = torch.clamp(rho_b, 1.65, 2.60)
    return vp, rho


def love_det_torch(
    f_hz: torch.Tensor,
    c_ms: torch.Tensor,
    h_m: torch.Tensor,
    vs_ms: torch.Tensor,
    rho_gcc: torch.Tensor,
) -> torch.Tensor:
    """Pole-free, overflow-free, differentiable Love-wave secular determinant D_L(f, c)."""
    omega = 2.0 * math.pi * f_hz
    k = omega / c_ms
    rho = rho_gcc * 1000.0
    mu = rho * vs_ms * vs_ms
    mu_ref = mu[-1]
    n_lay = vs_ms.shape[0] - 1

    q_hs = torch.sqrt(torch.clamp(1.0 - (c_ms / vs_ms[-1]) ** 2, min=1e-10))
    u = torch.ones_like(c_ms)
    t = -(mu[-1] / mu_ref) * q_hs

    for j in range(n_lay - 1, -1, -1):
        r2 = (c_ms / vs_ms[j]) ** 2 - 1.0
        kh = k * h_m[j]
        mu_r = mu[j] / mu_ref

        q_pos = torch.sqrt(torch.clamp(r2, min=1e-12))
        q_neg = torch.sqrt(torch.clamp(-r2, min=1e-12))

        ap = q_pos * kh
        cp = torch.cos(ap)
        # Taylor sinc fallback avoids 0/0 singularity at c == vs_j
        sp_div_q = torch.where(ap < 1e-5, kh * (1.0 - ap * ap / 6.0), torch.sin(ap) / q_pos)
        u_p = cp * u - (sp_div_q / mu_r) * t
        t_p = (mu_r * q_pos * torch.sin(ap)) * u + cp * t

        an = q_neg * kh
        th = torch.tanh(an)
        th_div_q = torch.where(an < 1e-5, kh * (1.0 - an * an / 3.0), th / q_neg)
        u_n = u - (th_div_q / mu_r) * t
        t_n = -(mu_r * q_neg * th) * u + t

        is_prop = r2 >= 0.0
        u_new = torch.where(is_prop, u_p, u_n)
        t_new = torch.where(is_prop, t_p, t_n)
        nrm = torch.sqrt(u_new * u_new + t_new * t_new + 1e-30)
        u = u_new / nrm
        t = t_new / nrm

    return t


def _compound_4x4(M: torch.Tensor) -> torch.Tensor:
    rows = []
    for r0, r1 in PAIRS_6:
        cols = []
        for c0, c1 in PAIRS_6:
            cols.append(M[:, r0, c0] * M[:, r1, c1] - M[:, r0, c1] * M[:, r1, c0])
        rows.append(torch.stack(cols, dim=-1))
    return torch.stack(rows, dim=-2)


def rayleigh_det_plucker_torch(
    f_hz: torch.Tensor,
    c_ms: torch.Tensor,
    h_m: torch.Tensor,
    vp_ms: torch.Tensor,
    vs_ms: torch.Tensor,
    rho_gcc: torch.Tensor,
) -> torch.Tensor:
    """Pole-free, overflow-free differentiable Rayleigh secular function D_R(f, c)
    via bounded-exponential 6-component Plucker compound propagator.
    """
    omega = 2.0 * math.pi * f_hz
    k = omega / c_ms
    rho = rho_gcc * 1000.0
    mu = rho * vs_ms * vs_ms
    mu_ref = mu[-1]
    n_lay = vs_ms.shape[0] - 1

    def layer_modes(j: int):
        mr = (mu[j] / mu_ref).to(torch.complex128)
        g = (2.0 - (c_ms / vs_ms[j]) ** 2).to(torch.complex128)
        arg_a = (1.0 - (c_ms / vp_ms[j]) ** 2).to(torch.complex128) + 1e-9j
        arg_b = (1.0 - (c_ms / vs_ms[j]) ** 2).to(torch.complex128) + 1e-9j
        ra = torch.sqrt(arg_a)
        rb = torch.sqrt(arg_b)
        ra = torch.where((ra.real < 0) | ((ra.real == 0) & (ra.imag < 0)), -ra, ra)
        rb = torch.where((rb.real < 0) | ((rb.real == 0) & (rb.imag < 0)), -rb, rb)
        one = torch.ones_like(g)
        E = torch.stack(
            [
                torch.stack([-one, rb, -one, -rb], dim=-1),
                torch.stack([-ra, one, ra, one], dim=-1),
                torch.stack([2.0 * mr * ra, -mr * g, -2.0 * mr * ra, -mr * g], dim=-1),
                torch.stack([mr * g, -2.0 * mr * rb, mr * g, 2.0 * mr * rb], dim=-1),
            ],
            dim=-2,
        )
        return E, ra, rb

    E_hs, _, _ = layer_modes(n_lay)
    C_hs = _compound_4x4(E_hs)
    v = C_hs[:, :, 0]
    v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-30)

    for j in range(n_lay - 1, -1, -1):
        Ej, ra, rb = layer_modes(j)
        Cj = _compound_4x4(Ej)
        alpha = torch.linalg.solve(Cj, v.unsqueeze(-1)).squeeze(-1)
        kh = (k * h_m[j]).to(torch.complex128)
        a = ra * kh
        b = rb * kh
        lam = torch.stack(
            [
                torch.ones_like(a),
                torch.exp(-(a + b)),
                torch.exp(-2.0 * b),
                torch.exp(-2.0 * a),
                torch.exp(-(a + b)),
                torch.exp(-2.0 * (a + b)),
            ],
            dim=-1,
        )
        v = torch.matmul(Cj, (lam * alpha).unsqueeze(-1)).squeeze(-1)
        v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-30)

    return v[:, 5]


def newton_velocity_residual_torch(
    det_fn: Callable[..., torch.Tensor],
    f_hz: torch.Tensor,
    c_obs: torch.Tensor,
    *model_args: torch.Tensor,
    eps_rel: float = 0.004,
) -> torch.Tensor:
    """Mode-number-free Newton-normalized velocity distance |D(f, c)| / |dD/dc| (m/s)."""
    dc = torch.clamp(c_obs * eps_rel, min=1.0)
    f_all = torch.cat([f_hz, f_hz, f_hz], dim=0)
    c_all = torch.cat([c_obs, c_obs + dc, c_obs - dc], dim=0)
    d_all = det_fn(f_all, c_all, *model_args)
    m = f_hz.shape[0]
    d0 = d_all[:m]
    dp = d_all[m : 2 * m]
    dm = d_all[2 * m :]

    slope = torch.abs(dp - dm) / (2.0 * dc)
    slope_floor = 1e-4 / torch.clamp(c_obs, min=100.0)
    err_c = torch.abs(d0) / torch.sqrt(slope * slope + slope_floor * slope_floor)
    cap = 0.45 * c_obs
    return cap * torch.tanh(err_c / cap)


def sh_transfer_peak_torch(
    h_m: torch.Tensor, vs_ms: torch.Tensor, rho_gcc: torch.Tensor, f_grid: torch.Tensor
) -> torch.Tensor:
    """Differentiable 1D viscoelastic SH transfer function amplitude |T(f)| in PyTorch."""
    xi = torch.where(vs_ms < 500.0, 0.025, torch.where(vs_ms < 900.0, 0.015, 0.008))
    vs_c = vs_ms.to(torch.complex128) * (1.0 + 1j * xi.to(torch.complex128))
    rho = (rho_gcc * 1000.0).to(torch.complex128)
    mu = rho * vs_c * vs_c
    omega = (2.0 * math.pi * f_grid).to(torch.complex128)

    u = torch.ones_like(omega)
    tau = torch.zeros_like(omega)
    for j in range(vs_ms.shape[0] - 1):
        kk = omega / vs_c[j]
        kh = kk * h_m[j].to(torch.complex128)
        c = torch.cos(kh)
        s = torch.sin(kh)
        zk = mu[j] * kk
        u_new = c * u + (s / zk) * tau
        tau_new = (-zk * s) * u + c * tau
        u, tau = u_new, tau_new

    k_hs = omega / vs_c[-1]
    z_hs = mu[-1] * k_hs
    return torch.abs(1.0 / (u + tau / (1j * z_hs)))


def invert_adsurf_joint(
    h_init_base: np.ndarray,
    vs_init_base: np.ndarray,
    f_disp: np.ndarray,
    c_r0_obs: np.ndarray,
    c_r0_std: Optional[np.ndarray] = None,
    valid_r0: Optional[np.ndarray] = None,
    r_hm_fc: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([])),
    c_l0_obs: Optional[np.ndarray] = None,
    c_l0_std: Optional[np.ndarray] = None,
    valid_l0: Optional[np.ndarray] = None,
    l_hm_fc: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([])),
    vs_ann_prior: Optional[np.ndarray] = None,
    z_grid: np.ndarray = Z_GRID_DEFAULT,
    alpha_love: Optional[np.ndarray] = None,
    pair_r_m: Optional[np.ndarray] = None,
    rho_zz_corr: Optional[np.ndarray] = None,
    rho_tt_re: Optional[np.ndarray] = None,
    f_rd: Optional[np.ndarray] = None,
    hvsr_obs_rd: Optional[np.ndarray] = None,
    raydec_obs: Optional[np.ndarray] = None,
    f0_obs: Optional[float] = None,
    hvsr_reliable: bool = True,
    d_max_m: float = 350.0,
    z_wt: float = 6.0,
    n_starts: int = 3,
    n_steps_stage1: int = 80,
    run_stage2: bool = True,
) -> Dict:
    """Two-stage mode-number-free ADsurf Rayleigh + Love + HVSR/SPAC joint inversion.

    Stage 1 runs multi-start PyTorch Adam on exact Rayleigh (Plucker) and Love (SH)
    secular determinants plus differentiable SH resonance frequency and smooth prior.
    Stage 2 optionally refines the top candidates via bounded Powell optimization
    adding DFA HVSR, RayDec ellipticity, and direct 3C SPAC spatial coherence.
    """
    n_tot = len(vs_init_base)
    n_fin = n_tot - 1
    if n_fin < 1:
        raise ValueError("Model must contain at least one finite layer plus a half-space.")

    if valid_r0 is None:
        valid_r0 = np.isfinite(c_r0_obs) & (c_r0_obs > 40.0)
    if c_r0_std is None:
        c_r0_std = 0.05 * np.where(valid_r0, c_r0_obs, 300.0)
    if c_l0_obs is None:
        c_l0_obs = np.full_like(f_disp, np.nan)
    if valid_l0 is None:
        valid_l0 = np.isfinite(c_l0_obs) & (c_l0_obs > 40.0)
    if c_l0_std is None:
        c_l0_std = 0.06 * np.where(valid_l0, c_l0_obs, 300.0)

    f_r_pts = list(f_disp[valid_r0])
    c_r_pts = list(c_r0_obs[valid_r0])
    s_r_pts = list(np.fmax(c_r0_std[valid_r0], 0.045 * c_r0_obs[valid_r0]))
    w_r_pts = [1.0] * len(f_r_pts)
    if len(r_hm_fc[0]) > 0:
        for fh, ch in zip(r_hm_fc[0], r_hm_fc[1]):
            f_r_pts.append(float(fh))
            c_r_pts.append(float(ch))
            s_r_pts.append(0.08 * float(ch))
            w_r_pts.append(0.45)

    f_l_pts = list(f_disp[valid_l0])
    c_l_pts = list(c_l0_obs[valid_l0])
    s_l_pts = list(np.fmax(c_l0_std[valid_l0], 0.055 * c_l0_obs[valid_l0]))
    w_l_pts = [0.85] * len(f_l_pts)
    if len(l_hm_fc[0]) > 0:
        for fh, ch in zip(l_hm_fc[0], l_hm_fc[1]):
            f_l_pts.append(float(fh))
            c_l_pts.append(float(ch))
            s_l_pts.append(0.09 * float(ch))
            w_l_pts.append(0.35)

    fr_t = torch.tensor(f_r_pts, dtype=torch.float64)
    cr_t = torch.tensor(c_r_pts, dtype=torch.float64)
    sr_t = torch.tensor(s_r_pts, dtype=torch.float64)
    wr_t = torch.tensor(w_r_pts, dtype=torch.float64)

    has_love = len(f_l_pts) >= 4
    if has_love:
        fl_t = torch.tensor(f_l_pts, dtype=torch.float64)
        cl_t = torch.tensor(c_l_pts, dtype=torch.float64)
        sl_t = torch.tensor(s_l_pts, dtype=torch.float64)
        wl_t = torch.tensor(w_l_pts, dtype=torch.float64)

    h0 = np.clip(h_init_base[:n_fin].astype(np.float64), 2.0, 360.0)
    vs0 = np.clip(vs_init_base.astype(np.float64), 90.0, 2600.0)
    tops0 = np.concatenate([[0.0], np.cumsum(h0)])
    z_mids0 = np.concatenate([tops0[:-1] + 0.5 * h0, [tops0[-1] + 60.0]])

    has_ann_prior = vs_ann_prior is not None
    if vs_ann_prior is None:
        iz0 = np.clip(np.searchsorted(tops0, z_grid, side="right") - 1, 0, n_tot - 1)
        vs_ann_prior = vs0[iz0].astype(np.float64)

    f_sh_t = torch.tensor(np.geomspace(0.20, 12.0, 96), dtype=torch.float64)
    log_f_sh_t = torch.log(f_sh_t)
    z_grid_t = torch.tensor(z_grid, dtype=torch.float64)
    log_vs_prior_t = torch.tensor(np.log(np.clip(vs_ann_prior, 80.0, 2600.0)), dtype=torch.float64)
    if has_ann_prior:
        w_prior_t = torch.tensor(np.where(z_grid <= d_max_m, 0.22, 1.4), dtype=torch.float64)
    else:
        w_prior_t = torch.full((len(z_grid),), 0.02, dtype=torch.float64)

    log_h_lo = np.log(np.clip(h0 * 0.35, 2.0, 300.0))
    log_h_hi = np.log(np.clip(h0 * 2.8, 12.0, 420.0))
    log_vs_lo = np.array([math.log(max(85.0, vs0[k] * (0.45 if z_mids0[k] <= d_max_m else 0.72))) for k in range(n_tot)])
    log_vs_hi = np.array([math.log(min(2650.0, vs0[k] * (2.20 if z_mids0[k] <= d_max_m else 1.45))) for k in range(n_tot)])

    lh_lo_t = torch.tensor(log_h_lo, dtype=torch.float64)
    lh_hi_t = torch.tensor(log_h_hi, dtype=torch.float64)
    lvs_lo_t = torch.tensor(log_vs_lo, dtype=torch.float64)
    lvs_hi_t = torch.tensor(log_vs_hi, dtype=torch.float64)

    rng = np.random.default_rng(202603)
    starts = [(np.log(h0), np.log(vs0))]
    for _ in range(max(0, n_starts - 1)):
        ph = np.clip(np.log(h0) + rng.normal(0.0, 0.25, size=n_fin), log_h_lo, log_h_hi)
        pvs = np.clip(np.log(vs0) + rng.normal(0.0, 0.16, size=n_tot), log_vs_lo, log_vs_hi)
        starts.append((ph, pvs))

    has_f0 = f0_obs is not None and np.isfinite(f0_obs) and f0_obs > 0.05
    w_f0 = (2.6 if hvsr_reliable else 0.35) if has_f0 else 0.0

    stage1_candidates = []
    for si, (lh_init, lvs_init) in enumerate(starts):
        lh = torch.tensor(lh_init, dtype=torch.float64, requires_grad=True)
        lvs = torch.tensor(lvs_init, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([lh, lvs], lr=0.055)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps_stage1, eta_min=0.004)

        best_loss = 1e18
        best_pair = (lh_init.copy(), lvs_init.copy())

        for _ in range(n_steps_stage1):
            opt.zero_grad()
            lh_c = torch.clamp(lh, lh_lo_t, lh_hi_t)
            lvs_c = torch.clamp(lvs, lvs_lo_t, lvs_hi_t)
            h_m = torch.cat([torch.exp(lh_c), torch.zeros(1, dtype=torch.float64)])
            vs_ms = torch.exp(lvs_c)
            vp_ms, rho_gcc = brocher_vp_rho_torch(h_m, vs_ms, z_wt=z_wt)

            err_r = newton_velocity_residual_torch(rayleigh_det_plucker_torch, fr_t, cr_t, h_m, vp_ms, vs_ms, rho_gcc)
            loss_r = torch.sum(wr_t * (err_r / sr_t) ** 2) / torch.sum(wr_t)

            if has_love:
                err_l = newton_velocity_residual_torch(love_det_torch, fl_t, cl_t, h_m, vs_ms, rho_gcc)
                loss_l = 0.85 * torch.sum(wl_t * (err_l / sl_t) ** 2) / torch.sum(wl_t)
            else:
                loss_l = torch.tensor(0.0, dtype=torch.float64)

            if has_f0 and f0_obs is not None:
                tf = sh_transfer_peak_torch(h_m, vs_ms, rho_gcc, f_sh_t)
                if hvsr_reliable:
                    band_m = (f_sh_t >= max(0.20, 0.45 * f0_obs)) & (f_sh_t <= min(12.0, 2.2 * f0_obs))
                else:
                    band_m = torch.ones_like(f_sh_t, dtype=torch.bool)
                w_soft = torch.softmax(6.0 * tf[band_m], dim=0)
                log_f0_soft = torch.sum(w_soft * log_f_sh_t[band_m])
                loss_f0 = w_f0 * ((log_f0_soft - math.log(max(f0_obs, 0.1))) / 0.12) ** 2
            else:
                loss_f0 = torch.tensor(0.0, dtype=torch.float64)

            tops_t = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(h_m[:-1], dim=0)])
            diffs = (z_grid_t[:, None] - tops_t[None, :]) / 4.0
            sig = torch.sigmoid(diffs)
            w_lay_cols = [1.0 - sig[:, 1]]
            for k in range(1, n_fin):
                w_lay_cols.append(sig[:, k] * (1.0 - sig[:, k + 1]))
            w_lay_cols.append(sig[:, n_fin])
            w_lay = torch.stack(w_lay_cols, dim=1)
            w_lay = w_lay / (w_lay.sum(dim=1, keepdim=True) + 1e-12)
            log_vs_g = torch.matmul(w_lay, lvs_c)
            loss_prior = torch.mean(w_prior_t * ((log_vs_g - log_vs_prior_t) / 0.35) ** 2)

            dv = lvs_c[1:] - lvs_c[:-1]
            loss_reg = (
                0.08 * torch.mean(dv * dv)
                + 1.6 * torch.sum(torch.relu(-dv - 0.32) ** 2)
                + 1.4 * torch.sum(torch.relu(dv - 0.90) ** 2)
            )

            loss = loss_r + loss_l + loss_f0 + loss_prior + loss_reg
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([lh, lvs], 4.0)
                opt.step()
                sched.step()
                with torch.no_grad():
                    lh.clamp_(lh_lo_t, lh_hi_t)
                    lvs.clamp_(lvs_lo_t, lvs_hi_t)
                lv = float(loss.item())
                if lv < best_loss:
                    best_loss = lv
                    best_pair = (lh.detach().numpy().copy(), lvs.detach().numpy().copy())

        h_c = np.concatenate([np.exp(best_pair[0]), [0.0]])
        vs_c = np.exp(best_pair[1])
        _, rho_c = brocher_vp_rho(h_c, vs_c, z_wt=z_wt)
        met_c = compute_profile_metrics(h_c, vs_c, rho_c, z_grid=z_grid)
        stage1_candidates.append(
            {
                "start_id": si,
                "stage1_loss": best_loss,
                "h_m": h_c,
                "vs_ms": vs_c,
                "z760_perm": met_c["z760_perm"] if float(met_c["has_z760"]) > 0.5 else np.nan,
            }
        )

    stage1_candidates.sort(key=lambda d: d["stage1_loss"])
    multistart_z760 = [float(c["z760_perm"]) for c in stage1_candidates if np.isfinite(c["z760_perm"])]

    if not run_stage2:
        h_opt = stage1_candidates[0]["h_m"]
        vs_opt = stage1_candidates[0]["vs_ms"]
        tops = np.zeros(n_tot)
        tops[1:] = np.cumsum(h_opt[:-1])
        vs_opt_grid = vs_opt[np.clip(np.searchsorted(tops, z_grid, side="right") - 1, 0, n_tot - 1)]
        return {
            "h_opt": h_opt,
            "vs_opt": vs_opt,
            "vs_opt_grid": vs_opt_grid,
            "stage1_loss": stage1_candidates[0]["stage1_loss"],
            "stage2_loss": stage1_candidates[0]["stage1_loss"],
            "multistart_z760_m": multistart_z760,
        }

    has_spac = (pair_r_m is not None) and (rho_zz_corr is not None)
    has_hvsr = (f_rd is not None) and (hvsr_obs_rd is not None) and len(f_rd) >= 16
    has_raydec = has_hvsr and (raydec_obs is not None)

    if has_hvsr and f_rd is not None and hvsr_obs_rd is not None:
        f_rd_sub_idx = np.linspace(2, len(f_rd) - 4, min(14, len(f_rd) - 4), dtype=int)
        f_rd_sub = f_rd[f_rd_sub_idx]
        hv_obs_sub = hvsr_obs_rd[f_rd_sub_idx]
        rd_obs_sub = raydec_obs[f_rd_sub_idx] if (has_raydec and raydec_obs is not None) else None
    else:
        f_rd_sub = hv_obs_sub = rd_obs_sub = None

    valid_r_idx = np.where(valid_r0)[0]
    valid_l_idx = np.where(valid_l0)[0]
    spac_r_idx = valid_r_idx[:: max(1, len(valid_r_idx) // 9)]
    spac_l_idx = valid_l_idx[:: max(1, len(valid_l_idx) // 7)] if len(valid_l_idx) > 0 else np.array([], dtype=int)
    f_sh_np = np.geomspace(0.20, 14.0, 112)
    bounds_p = list(zip(log_h_lo, log_h_hi)) + list(zip(log_vs_lo, log_vs_hi))
    log_vs_prior_np = np.log(np.clip(vs_ann_prior, 80.0, 2600.0))
    w_prior_np = np.where(z_grid <= d_max_m, 0.20, 1.4) if has_ann_prior else np.full(len(z_grid), 0.02, dtype=np.float64)

    def stage2_objective(x: np.ndarray) -> float:
        h_m = np.concatenate([np.exp(x[:n_fin]), [0.0]])
        vs_ms = np.exp(x[n_fin:])
        vp_ms, rho_gcc = brocher_vp_rho(h_m, vs_ms, z_wt=z_wt)

        with torch.no_grad():
            ht = torch.tensor(h_m, dtype=torch.float64)
            vpt = torch.tensor(vp_ms, dtype=torch.float64)
            vst = torch.tensor(vs_ms, dtype=torch.float64)
            rhot = torch.tensor(rho_gcc, dtype=torch.float64)
            er = newton_velocity_residual_torch(rayleigh_det_plucker_torch, fr_t, cr_t, ht, vpt, vst, rhot)
            loss_det_r = float((torch.sum(wr_t * (er / sr_t) ** 2) / torch.sum(wr_t)).item())
            if has_love:
                el = newton_velocity_residual_torch(love_det_torch, fl_t, cl_t, ht, vst, rhot)
                loss_det_l = 0.85 * float((torch.sum(wl_t * (el / sl_t) ** 2) / torch.sum(wl_t)).item())
            else:
                loss_det_l = 0.0

        vm = np.column_stack([h_m / 1000.0, vp_ms / 1000.0, vs_ms / 1000.0, rho_gcc])
        loss_spac = 0.0
        if has_spac and pair_r_m is not None and rho_zz_corr is not None:
            try:
                pd_c = PhaseDispersion(*vm.T)
                t_r_s = (1.0 / f_disp[spac_r_idx])[::-1].copy()
                r0_s = pd_c(t_r_s, mode=0, wave="rayleigh")
                if len(r0_s.velocity) == len(t_r_s):
                    c_r_s = (r0_s.velocity * 1000.0)[::-1]
                    sq = []
                    for kk, fi in enumerate(spac_r_idx):
                        kr = 2.0 * np.pi * f_disp[fi] * pair_r_m / c_r_s[kk]
                        ps = (kr >= 0.15) & (kr <= 11.0)
                        if ps.sum() < min(4, len(pair_r_m)):
                            ps = np.argsort(pair_r_m)[: min(8, len(pair_r_m))]
                        sq.append(np.mean((rho_zz_corr[ps, fi] - special.j0(kr[ps])) ** 2))
                    loss_spac += 1.6 * float(np.mean(sq)) / (0.14**2)
                if (
                    len(spac_l_idx) >= 3
                    and rho_tt_re is not None
                    and alpha_love is not None
                ):
                    t_l_s = (1.0 / f_disp[spac_l_idx])[::-1].copy()
                    l0_s = pd_c(t_l_s, mode=0, wave="love")
                    if len(l0_s.velocity) == len(t_l_s):
                        c_l_s = (l0_s.velocity * 1000.0)[::-1]
                        sq_t = []
                        for kk, fi in enumerate(spac_l_idx):
                            kl = 2.0 * np.pi * f_disp[fi] * pair_r_m / c_l_s[kk]
                            kr = 2.0 * np.pi * f_disp[fi] * pair_r_m / max(c_r0_obs[fi], 80.0)
                            ps = (kl >= 0.15) & (kl <= 11.0)
                            if ps.sum() < min(4, len(pair_r_m)):
                                ps = np.argsort(pair_r_m)[: min(8, len(pair_r_m))]
                            al = float(alpha_love[fi])
                            mod_tt = al * (special.j0(kl[ps]) - special.jv(2, kl[ps])) + (1.0 - al) * (
                                special.j0(kr[ps]) + special.jv(2, kr[ps])
                            )
                            sq_t.append(np.mean((rho_tt_re[ps, fi] - mod_tt) ** 2))
                        loss_spac += 0.9 * float(np.mean(sq_t)) / (0.18**2)
            except Exception:
                loss_spac += 4.0

        loss_f0 = 0.0
        if has_f0 and f0_obs is not None:
            f0_sh, _ = sh_transfer_f0(h_m, vs_ms, rho_gcc, f_sh_np)
            loss_f0 = w_f0 * ((math.log(max(f0_sh, 0.1)) - math.log(max(f0_obs, 0.1))) / 0.12) ** 2

        loss_dfa = 0.0
        if has_hvsr and f_rd_sub is not None and hv_obs_sub is not None:
            dfa_sub, _, _, _ = compute_dfa_hvsr_full(
                h_m, vp_ms, vs_ms, rho_gcc, f_rd_sub, damping=0.012, n_k_core=220, n_k_tail=24
            )
            w_dfa = 1.4 if hvsr_reliable else 0.35
            loss_dfa = w_dfa * float(
                np.mean(((np.log10(np.maximum(dfa_sub, 0.1)) - np.log10(np.maximum(hv_obs_sub, 0.1))) / 0.18) ** 2)
            )

        loss_raydec = 0.0
        if rd_obs_sub is not None and f_rd_sub is not None:
            try:
                t_rd_sub_asc = (1.0 / f_rd_sub)[::-1].copy()
                el_s = Ellipticity(*vm.T)(t_rd_sub_asc, mode=0)
                if len(el_s.period) >= 5:
                    ix_el = np.clip(np.searchsorted(t_rd_sub_asc, el_s.period), 0, len(f_rd_sub) - 1)
                    ok_el = np.abs(t_rd_sub_asc[ix_el] - el_s.period) < (1e-4 * el_s.period + 1e-6)
                    el_mod = np.full(len(f_rd_sub), np.nan)
                    el_mod[(len(f_rd_sub) - 1) - ix_el[ok_el]] = np.abs(el_s.ellipticity[ok_el])
                    m_rd = np.isfinite(el_mod) & np.isfinite(rd_obs_sub) & (el_mod > 0.25) & (el_mod < 15.0)
                    if m_rd.sum() >= 5:
                        w_rd = 0.55 if hvsr_reliable else 0.20
                        loss_raydec = w_rd * float(
                            np.mean(
                                (
                                    (
                                        np.log10(np.clip(el_mod[m_rd], 0.30, 12.0))
                                        - np.log10(np.clip(rd_obs_sub[m_rd], 0.30, 12.0))
                                    )
                                    / 0.25
                                )
                                ** 2
                            )
                        )
            except Exception:
                pass

        tops = np.zeros(n_tot)
        tops[1:] = np.cumsum(h_m[:-1])
        iz = np.clip(np.searchsorted(tops, z_grid, side="right") - 1, 0, n_tot - 1)
        loss_prior = float(np.mean(w_prior_np * ((np.log(vs_ms[iz]) - log_vs_prior_np) / 0.35) ** 2))
        dv = np.diff(np.log(vs_ms))
        loss_reg = (
            0.08 * float(np.mean(dv * dv))
            + 1.6 * float(np.sum(np.maximum(0.0, -dv - 0.32) ** 2))
            + 1.4 * float(np.sum(np.maximum(0.0, dv - 0.90) ** 2))
        )

        return loss_det_r + loss_det_l + loss_spac + loss_f0 + loss_dfa + loss_raydec + loss_prior + loss_reg

    best_x = np.concatenate([np.log(stage1_candidates[0]["h_m"][:-1]), np.log(stage1_candidates[0]["vs_ms"])])
    best_val = stage2_objective(best_x)

    for cand in stage1_candidates[: min(2, len(stage1_candidates))]:
        x_s = np.concatenate([np.log(cand["h_m"][:-1]), np.log(cand["vs_ms"])])
        res = optimize.minimize(
            stage2_objective,
            x_s,
            method="Powell",
            bounds=bounds_p,
            options={"maxiter": 20, "maxfev": 350, "ftol": 2.5e-3, "xtol": 2.0e-2},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = res.x.copy()

    h_opt = np.concatenate([np.exp(best_x[:n_fin]), [0.0]])
    vs_opt = np.exp(best_x[n_fin:])
    tops = np.zeros(n_tot)
    tops[1:] = np.cumsum(h_opt[:-1])
    vs_opt_grid = vs_opt[np.clip(np.searchsorted(tops, z_grid, side="right") - 1, 0, n_tot - 1)]

    return {
        "h_opt": h_opt,
        "vs_opt": vs_opt,
        "vs_opt_grid": vs_opt_grid,
        "stage1_loss": stage1_candidates[0]["stage1_loss"],
        "stage2_loss": best_val,
        "multistart_z760_m": multistart_z760,
        "multistart_z760_std_m": float(np.std(multistart_z760, ddof=1)) if len(multistart_z760) >= 2 else 0.0,
        "multistart_z760_range_m": float(np.max(multistart_z760) - np.min(multistart_z760)) if len(multistart_z760) >= 2 else 0.0,
    }
