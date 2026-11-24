"""3-component array processing: near-DC thermal drift removal, Sleeman 3-station
incoherent self-noise diagnostics, horizontal E/N -> Radial/Transverse rotation,
3C SPAC coherency (ZZ, RR, TT, real-part antisymmetric ZR/RZ), and Hu et al. (2020)
decoupled Hankel Frequency-Bessel (3C F-J) dispersion analysis.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal, special

from .physics import F_GRID_DEFAULT


def konno_ohmachi_matrix(f_src: np.ndarray, f_tgt: np.ndarray, b: float = 35.0) -> np.ndarray:
    """Build Konno-Ohmachi smoothing weight matrix from f_src to f_tgt."""
    f_src_safe = np.maximum(f_src, 1e-12)[None, :]
    f_tgt_safe = np.maximum(f_tgt, 1e-12)[:, None]
    ratio = f_src_safe / f_tgt_safe
    x = b * np.log10(ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(x) < 1e-8, 1.0, (np.sin(x) / x) ** 4)
    w[np.abs(np.log10(ratio)) > (3.5 / b)] = 0.0
    w[:, f_src <= 0] = 0.0
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return w / row_sum


def ehz_4p5hz_theoretical_gain_db(
    freqs: np.ndarray, fn: float = 4.5, damping: float = 0.70
) -> np.ndarray:
    """Theoretical velocity amplitude response (dB re plateau) of a 4.5 Hz nodal geophone."""
    u = freqs / fn
    mag = (u**2) / np.sqrt((1.0 - u**2) ** 2 + (2.0 * damping * u) ** 2)
    return 20.0 * np.log10(np.maximum(mag, 1e-12))


def remove_near_dc_drift(
    trace: np.ndarray, fs: float, cutoff_hz: float = 0.15
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Remove near-DC thermal/tilt drift using linear detrend + 4th-order zero-phase
    Butterworth high-pass filter. Returns cleaned trace and pre/post drift energy diagnostics.
    """
    raw_d = signal.detrend(trace, type="linear")
    nper = min(len(raw_d), int(round(120.0 * fs)))
    fp, p_raw = signal.welch(raw_d, fs=fs, nperseg=nper)
    m_dc = (fp >= 0.02) & (fp < cutoff_hz)
    m_mid = (fp >= 0.50) & (fp <= 4.0)
    p_dc_pre = float(np.mean(p_raw[m_dc])) if m_dc.any() else 0.0
    p_mid_pre = float(np.mean(p_raw[m_mid])) if m_mid.any() else 1e-30

    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=fs, output="sos")
    clean = signal.sosfiltfilt(sos, raw_d)

    _, p_cln = signal.welch(clean, fs=fs, nperseg=nper)
    p_dc_post = float(np.mean(p_cln[m_dc])) if m_dc.any() else 0.0
    p_mid_post = float(np.mean(p_cln[m_mid])) if m_mid.any() else 1e-30

    diag = {
        "dc_to_mid_ratio_pre": p_dc_pre / max(p_mid_pre, 1e-30),
        "dc_to_mid_ratio_post": p_dc_post / max(p_mid_post, 1e-30),
        "drift_suppression_db": float(10.0 * np.log10(max(p_dc_pre, 1e-30) / max(p_dc_post, 1e-30))),
    }
    return clean, diag


def rotate_pair_en_to_rt(
    e1: np.ndarray,
    n1: np.ndarray,
    e2: np.ndarray,
    n2: np.ndarray,
    az_rad: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate horizontal East/North traces (or spectra) of station pair (1->2) at azimuth az_rad
    into Radial (R along pair axis) and Transverse (T perpendicular, 90 deg CW from R).
    """
    sa = float(np.sin(az_rad))
    ca = float(np.cos(az_rad))
    r1 = e1 * sa + n1 * ca
    r2 = e2 * sa + n2 * ca
    t1 = e1 * ca - n1 * sa
    t2 = e2 * ca - n2 * sa
    return r1, r2, t1, t2


def estimate_sleeman_nsr(
    szz_sm: np.ndarray,
    tri: Tuple[int, int, int] = (0, 1, 2),
) -> np.ndarray:
    """Estimate 3-station Sleeman (2006) incoherent self-noise-to-signal ratio (NSR)
    on smoothed cross-spectral matrix szz_sm of shape (n_sta, n_sta, n_f).
    """
    i0, j0, k0 = tri
    nsr_list = []
    for a, b, c in ((i0, j0, k0), (j0, i0, k0), (k0, i0, j0)):
        coh_bc = np.abs(szz_sm[b, c]) / np.sqrt(
            np.maximum(np.real(szz_sm[b, b]) * np.real(szz_sm[c, c]), 1e-30)
        )
        denom = np.where(np.abs(szz_sm[b, c]) > 1e-20, np.conj(szz_sm[b, c]), 1e-20)
        sig = np.real(szz_sm[a, b] * np.conj(szz_sm[a, c]) / denom)
        tot = np.maximum(np.real(szz_sm[a, a]), 1e-30)
        nz = np.where(
            (coh_bc > 0.25) & (sig > 0.05 * tot),
            np.clip((tot - sig) / np.maximum(sig, 1e-30), 0.0, 20.0),
            np.nan,
        )
        nsr_list.append(nz)
    with np.errstate(all="ignore"):
        return np.nanmedian(np.vstack(nsr_list), axis=0)


def estimate_array_nsr_and_correct(
    r_arr: np.ndarray,
    rho_zz_re: np.ndarray,
    rho_zz_im: np.ndarray,
    rho_zz_se: np.ndarray,
    c_base: np.ndarray,
    f_grid: np.ndarray = F_GRID_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Estimate zero-lag coherence factor A(f) = 1 / (1 + NSR(f)) by fitting
    rho_obs(f, r) = A(f) * J0(k*r) on short-to-medium pairs (kr <= 4.5).
    Applies conservative NSR recovery ONLY in A in [0.45, 0.96]; never amplifies noise-dominated bands.
    """
    _, n_f = rho_zz_re.shape
    a_factor = np.ones(n_f, dtype=np.float64)
    a_se = np.zeros(n_f, dtype=np.float64)
    rho_corr = rho_zz_re.copy()

    for fi, f in enumerate(f_grid):
        c0 = float(c_base[fi])
        kr = 2.0 * np.pi * f * r_arr / max(c0, 80.0)
        j0 = special.j0(kr)
        sel = (kr <= 4.5) & (np.abs(j0) >= 0.22)
        if sel.sum() < 4:
            sel = np.argsort(r_arr)[: min(6, len(r_arr))]
        y = rho_zz_re[sel, fi]
        x = j0[sel]
        w = 1.0 / (rho_zz_se[sel, fi] ** 2 + 0.04**2 + 0.25 * rho_zz_im[sel, fi] ** 2)
        num = float(np.sum(w * x * y))
        den = float(np.sum(w * x * x)) + 1e-12
        a_hat = float(np.clip(num / den, 0.05, 1.05))
        a_cl = float(np.clip(a_hat, 0.05, 1.0))
        resid = y - a_hat * x
        var_a = float(np.sum(w * resid * resid) / max(int(sel.sum()) - 1, 1)) / den
        a_factor[fi] = a_cl
        a_se[fi] = float(np.sqrt(max(var_a, 0.0)))

        if 0.45 <= a_cl <= 0.96:
            rho_corr[:, fi] = np.clip(rho_zz_re[:, fi] / a_cl, -1.0, 1.0)

    nsr_eff = np.clip(1.0 / np.maximum(a_factor, 0.05) - 1.0, 0.0, 19.0)
    snr_db = 10.0 * np.log10(1.0 / np.maximum(nsr_eff, 1e-3))
    recoverable = a_factor >= 0.40
    noise_dominated = a_factor < 0.35

    mid_mask = (f_grid >= 1.0) & (f_grid <= 4.5)
    mid_idx = np.where(mid_mask)[0] if np.any(mid_mask) else np.arange(n_f)
    anchor_fi = int(mid_idx[int(np.argmax(a_factor[mid_idx]))])
    f_min_nsr = float(f_grid[0])
    for fi in range(anchor_fi, -1, -1):
        if a_factor[fi] < 0.35 and (fi == 0 or a_factor[max(0, fi - 1)] < 0.42):
            f_min_nsr = float(f_grid[min(fi + 1, n_f - 1)])
            break

    return {
        "a_factor": a_factor,
        "a_se": a_se,
        "nsr_eff": nsr_eff,
        "snr_db": snr_db,
        "recoverable": recoverable,
        "noise_dominated": noise_dominated,
        "f_min_nsr_hz": f_min_nsr,
        "rho_zz_corr": rho_corr,
    }


def compute_3c_pair_coherence(
    spec_e1: np.ndarray,
    spec_n1: np.ndarray,
    spec_z1: np.ndarray,
    spec_e2: np.ndarray,
    spec_n2: np.ndarray,
    spec_z2: np.ndarray,
    az_rad: float,
    ko_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute Konno-Ohmachi smoothed 3C complex coherencies (ZZ, RR, TT, and real-part
    antisymmetric ZR/RZ) across windows for one station pair at azimuth az_rad (1->2).
    """
    r1, r2, t1, t2 = rotate_pair_en_to_rt(spec_e1, spec_n1, spec_e2, spec_n2, az_rad)

    def smooth_coh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sab = ko_mat @ np.mean(a * np.conj(b), axis=0)
        saa = ko_mat @ np.mean(np.abs(a) ** 2, axis=0)
        sbb = ko_mat @ np.mean(np.abs(b) ** 2, axis=0)
        return sab / np.sqrt(np.maximum(saa * sbb, 1e-30))

    rho_zz = smooth_coh(spec_z1, spec_z2)
    rho_rr = smooth_coh(r1, r2)
    rho_tt = smooth_coh(t1, t2)
    rho_zr = smooth_coh(spec_z1, r2)
    rho_rz = smooth_coh(r1, spec_z2)

    return {
        "rho_zz_re": np.real(rho_zz),
        "rho_zz_im": np.imag(rho_zz),
        "rho_rr_re": np.real(rho_rr),
        "rho_rr_im": np.imag(rho_rr),
        "rho_tt_re": np.real(rho_tt),
        "rho_tt_im": np.imag(rho_tt),
        # Isotropic Rayleigh R is +-i relative to Z, and azimuthal average of cos(theta)*exp(ikr cos(theta))
        # introduces a second factor of i (i*J1(kr)), making <Z1 R2*> = +-E*J1(kr) purely REAL and antisymmetric.
        "rho_zr_anti": 0.5 * (np.real(rho_zr) - np.real(rho_rz)),
    }


def _viterbi_track(
    score: np.ndarray,
    c_grid: np.ndarray,
    f_grid: np.ndarray,
    c_guide: np.ndarray | None = None,
    guide_lo_ratio: float = 0.85,
    guide_hi_ratio: float = 1.65,
) -> np.ndarray:
    """Normal-dispersion-biased dynamic programming ridge tracker."""
    n_f, n_c = score.shape
    dlogf = np.diff(np.log(f_grid))
    log_c = np.log(c_grid)
    dp = np.full((n_f, n_c), -1e9, dtype=np.float64)
    ptr = np.zeros((n_f, n_c), dtype=np.int32)

    prior = np.zeros((n_f, n_c), dtype=np.float64)
    for fi, f in enumerate(f_grid):
        c_max_soft = min(2800.0, 2300.0 / (f**0.52) + 380.0)
        c_min_soft = max(110.0, 130.0 / (f**0.25))
        prior[fi] -= 0.4 * np.maximum(0.0, (c_grid - c_max_soft) / 300.0) ** 2
        prior[fi] -= 0.4 * np.maximum(0.0, (c_min_soft - c_grid) / 100.0) ** 2
        if c_guide is not None and np.isfinite(c_guide[fi]):
            c_lo = c_guide[fi] * guide_lo_ratio
            c_hi = c_guide[fi] * guide_hi_ratio
            prior[fi] -= 0.7 * np.maximum(0.0, (c_lo - c_grid) / (0.15 * c_guide[fi])) ** 2
            prior[fi] -= 0.5 * np.maximum(0.0, (c_grid - c_hi) / (0.20 * c_guide[fi])) ** 2

    dp[0] = score[0] + prior[0]
    for fi in range(1, n_f):
        dlf = dlogf[fi - 1]
        slope = (log_c[None, :] - log_c[:, None]) / dlf
        pen = (
            0.09 * (np.clip(slope + 0.35, -3.0, 3.0)) ** 2
            + 1.9 * np.maximum(0.0, slope - 0.16) ** 2
            + 0.9 * np.maximum(0.0, -1.6 - slope) ** 2
        )
        pen[np.abs(log_c[None, :] - log_c[:, None]) > 0.35] += 5.0
        vals = dp[fi - 1][:, None] - pen
        best_prev = np.argmax(vals, axis=0)
        dp[fi] = vals[best_prev, np.arange(n_c)] + score[fi] + prior[fi]
        ptr[fi] = best_prev

    idx = np.zeros(n_f, dtype=np.int32)
    idx[-1] = int(np.argmax(dp[-1]))
    for fi in range(n_f - 2, -1, -1):
        idx[fi] = ptr[fi + 1, idx[fi + 1]]
    return idx


def compute_decoupled_3c_fj(
    freqs: np.ndarray,
    c_grid: np.ndarray,
    pair_r_m: np.ndarray,
    rho_zz: np.ndarray,
    rho_rr: np.ndarray,
    rho_tt: np.ndarray,
    rho_zr_anti: np.ndarray | None = None,
    c_r0_guide: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Compute Hu et al. (2020) decoupled 3C Frequency-Bessel (F-J) spectra for:
      - Z-Z Rayleigh: J0 kernel
      - Pure Love (T-T / R-R decoupled): 0.25 * [rho_TT*(J0-J2) + rho_RR*(J0+J2)]
      - Pure Radial Rayleigh (R-R / T-T decoupled): 0.25 * [rho_RR*(J0-J2) + rho_TT*(J0+J2)]
      - Cross Z-R Rayleigh: J1 kernel on real antisymmetric rho_zr_anti
    """
    n_f = len(freqs)
    n_c = len(c_grid)
    order = np.argsort(pair_r_m)
    r_sorted = pair_r_m[order]
    dr = np.zeros_like(r_sorted)
    if len(r_sorted) > 1:
        dr[0] = 0.5 * (r_sorted[1] - r_sorted[0]) + 0.5 * r_sorted[0]
        dr[-1] = 0.5 * (r_sorted[-1] - r_sorted[-2])
        dr[1:-1] = 0.5 * (r_sorted[2:] - r_sorted[:-2])
    dr = np.maximum(dr, 0.5)
    r_taper = np.exp(-0.25 * (r_sorted / max(float(r_sorted[-1]), 1.0)) ** 2)
    w_fj = r_sorted * dr * r_taper

    fj_zz_norm = np.zeros((n_f, n_c), dtype=np.float64)
    fj_love_norm = np.zeros((n_f, n_c), dtype=np.float64)
    fj_rr_norm = np.zeros((n_f, n_c), dtype=np.float64)
    fj_zr_norm = np.zeros((n_f, n_c), dtype=np.float64)

    rho_zz_s = rho_zz[order]
    rho_rr_s = rho_rr[order]
    rho_tt_s = rho_tt[order]
    rho_zr_s = rho_zr_anti[order] if rho_zr_anti is not None else None

    for fi, f in enumerate(freqs):
        arg = (2.0 * np.pi * f) * (r_sorted[:, None] / c_grid[None, :])
        j0 = special.j0(arg)
        j1 = special.j1(arg)
        j2 = special.jv(2, arg)
        km = j0 - j2
        kp = j0 + j2

        fz = np.sum((rho_zz_s[:, fi] * w_fj)[:, None] * j0, axis=0)
        fj_zz_norm[fi] = fz / (np.max(np.abs(fz)) + 1e-12)

        fl = 0.25 * np.sum(w_fj[:, None] * (rho_tt_s[:, fi : fi + 1] * km + rho_rr_s[:, fi : fi + 1] * kp), axis=0)
        fr = 0.25 * np.sum(w_fj[:, None] * (rho_rr_s[:, fi : fi + 1] * km + rho_tt_s[:, fi : fi + 1] * kp), axis=0)
        fj_love_norm[fi] = fl / (np.max(np.abs(fl)) + 1e-12)
        fj_rr_norm[fi] = fr / (np.max(np.abs(fr)) + 1e-12)

        if rho_zr_s is not None:
            fzr = np.sum((rho_zr_s[:, fi] * w_fj)[:, None] * j1, axis=0)
            fj_zr_norm[fi] = fzr / (np.max(np.abs(fzr)) + 1e-12)

    idx_l = _viterbi_track(
        np.clip(fj_love_norm, 0.0, 1.0),
        c_grid,
        freqs,
        c_guide=c_r0_guide,
        guide_lo_ratio=0.45,
        guide_hi_ratio=1.60,
    )
    idx_rr = _viterbi_track(
        np.clip(fj_rr_norm, 0.0, 1.0),
        c_grid,
        freqs,
        c_guide=c_r0_guide,
        guide_lo_ratio=0.75,
        guide_hi_ratio=1.30,
    )

    return {
        "fj_zz_norm": fj_zz_norm,
        "fj_love_norm": fj_love_norm,
        "fj_rr_norm": fj_rr_norm,
        "fj_zr_norm": fj_zr_norm,
        "c_l0_track": c_grid[idx_l],
        "c_rr_track": c_grid[idx_rr],
    }
