"""Pure Rayleigh-wave ellipticity extraction via Hobiger et al. (2009) RayDec
and Full Elastic Diffuse-Field Assumption (DFA) HVSR forward modeling
(Sanchez-Sesma et al. 2011 / Garcia-Jerez et al. 2016) via Kennett-Luco-Apsel
bounded-exponential R/T Green's functions (Numba-accelerated).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numba import njit
from scipy import signal


@njit(cache=False)
def _psv_eig(k: float, omega_c: complex, vp: float, vs: float, rho: float):
    """Elastic P-SV eigenmatrices at complex frequency omega_c = omega*(1 - i*eps)."""
    kp = omega_c / vp
    ks = omega_c / vs
    mu = rho * vs * vs
    nu_p = np.sqrt(k * k - kp * kp)
    nu_s = np.sqrt(k * k - ks * ks)
    if nu_p.real < 0.0 or (nu_p.real == 0.0 and nu_p.imag < 0.0):
        nu_p = -nu_p
    if nu_s.real < 0.0 or (nu_s.real == 0.0 and nu_s.imag < 0.0):
        nu_s = -nu_s

    g = 2.0 * k * k - ks * ks
    s = 1.0 / (abs(ks) + 1e-12)
    Md = np.empty((2, 2), dtype=np.complex128)
    Mu = np.empty((2, 2), dtype=np.complex128)
    Nd = np.empty((2, 2), dtype=np.complex128)
    Nu = np.empty((2, 2), dtype=np.complex128)

    Md[0, 0] = -k * s
    Md[0, 1] = nu_s * s
    Md[1, 0] = -nu_p * s
    Md[1, 1] = k * s

    Mu[0, 0] = -k * s
    Mu[0, 1] = -nu_s * s
    Mu[1, 0] = nu_p * s
    Mu[1, 1] = k * s

    Nd[0, 0] = 2.0 * mu * k * nu_p * s
    Nd[0, 1] = -mu * g * s
    Nd[1, 0] = mu * g * s
    Nd[1, 1] = -2.0 * mu * k * nu_s * s

    Nu[0, 0] = -2.0 * mu * k * nu_p * s
    Nu[0, 1] = -mu * g * s
    Nu[1, 0] = mu * g * s
    Nu[1, 1] = 2.0 * mu * k * nu_s * s
    return Md, Mu, Nd, Nu, nu_p, nu_s


@njit(cache=False)
def _inv2(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    out = np.empty((2, 2), dtype=np.complex128)
    out[0, 0] = A[1, 1] / det
    out[0, 1] = -A[0, 1] / det
    out[1, 0] = -A[1, 0] / det
    out[1, 1] = A[0, 0] / det
    return out


@njit(cache=False)
def _mul2(A, B):
    out = np.empty((2, 2), dtype=np.complex128)
    out[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    out[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    out[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    out[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    return out


@njit(cache=False)
def _dfa_green_integrand(k: float, omega_c: complex, h_m, vp_ms, vs_ms, rho):
    """Exact elastic bounded-exponential R/T surface impedance recursion at complex frequency omega_c."""
    n_tot = len(vs_ms)
    n_lay = n_tot - 1

    # SH (Love + SH body waves)
    mu_hs = rho[n_lay] * vs_ms[n_lay] * vs_ms[n_lay]
    ks_hs = omega_c / vs_ms[n_lay]
    nu_hs = np.sqrt(k * k - ks_hs * ks_hs)
    if nu_hs.real < 0.0:
        nu_hs = -nu_hs
    Z_sh = mu_hs * nu_hs
    for j in range(n_lay - 1, -1, -1):
        mu_j = rho[j] * vs_ms[j] * vs_ms[j]
        ks_j = omega_c / vs_ms[j]
        nu_j = np.sqrt(k * k - ks_j * ks_j)
        if nu_j.real < 0.0:
            nu_j = -nu_j
        zj = mu_j * nu_j
        e2 = np.exp(-2.0 * nu_j * h_m[j])
        th = (1.0 - e2) / (1.0 + e2)
        Z_sh = zj * (Z_sh + zj * th) / (zj + Z_sh * th)
    g_sh = 1.0 / Z_sh

    # P-SV (Rayleigh + P/SV body waves)
    Md, _, Nd, _, _, _ = _psv_eig(k, omega_c, vp_ms[n_lay], vs_ms[n_lay], rho[n_lay])
    Z = _mul2(-Nd, _inv2(Md))
    for j in range(n_lay - 1, -1, -1):
        Md, Mu, Nd, Nu, nu_p, nu_s = _psv_eig(k, omega_c, vp_ms[j], vs_ms[j], rho[j])
        ep = np.exp(-nu_p * h_m[j])
        es = np.exp(-nu_s * h_m[j])
        A = Nu + _mul2(Z, Mu)
        B = Nd + _mul2(Z, Md)
        Rb = _mul2(-_inv2(A), B)
        Me = np.empty((2, 2), dtype=np.complex128)
        Me[0, 0] = ep * Rb[0, 0] * ep
        Me[0, 1] = ep * Rb[0, 1] * es
        Me[1, 0] = es * Rb[1, 0] * ep
        Me[1, 1] = es * Rb[1, 1] * es
        U_top = Md + _mul2(Mu, Me)
        T_top = -Nd - _mul2(Nu, Me)
        Z = _mul2(T_top, _inv2(U_top))

    G = _inv2(Z)
    return G[0, 0], G[1, 1], g_sh


@njit(cache=False)
def compute_dfa_hvsr_full(
    h_m: np.ndarray,
    vp_ms: np.ndarray,
    vs_ms: np.ndarray,
    rho_gcc: np.ndarray,
    freqs: np.ndarray,
    damping: float = 0.012,
    n_k_core: int = 360,
    n_k_tail: int = 36,
):
    """Exact Elastic Diffuse-Field Assumption (DFA) HVSR (Sanchez-Sesma et al. 2011 /
    Garcia-Jerez et al. 2016) including Rayleigh + Love + P/SV/SH body waves.
    Returns (hv_rms, hv_tot, hv_psv_rms, love_energy_fraction).
    """
    n_f = len(freqs)
    hv_rms = np.zeros(n_f, dtype=np.float64)
    hv_tot = np.zeros(n_f, dtype=np.float64)
    hv_psv_rms = np.zeros(n_f, dtype=np.float64)
    love_frac = np.zeros(n_f, dtype=np.float64)

    rho = rho_gcc * 1000.0
    vs_min = np.min(vs_ms)
    vs_max = np.max(vs_ms)

    u_lo = max(0.004, 0.10 * (vs_min / max(vs_max, vs_min)))
    n_k = n_k_core + n_k_tail
    u_grid = np.empty(n_k, dtype=np.float64)
    log_lo = np.log(u_lo)
    log_hi = np.log(1.38)
    for i in range(n_k):
        u_grid[i] = np.exp(log_lo + (log_hi - log_lo) * i / (n_k - 1))

    for fi in range(n_f):
        omega = 2.0 * np.pi * freqs[fi]
        omega_c = omega * (1.0 - 1j * damping)
        k_scale = omega / vs_min
        int_rr = 0.0
        int_zz = 0.0
        int_sh = 0.0
        for ik in range(n_k):
            u = u_grid[ik]
            if ik == 0:
                du = u_grid[1] - u_grid[0]
            elif ik == n_k - 1:
                du = u_grid[ik] - u_grid[ik - 1]
            else:
                du = 0.5 * (u_grid[ik + 1] - u_grid[ik - 1])
            k = u * k_scale
            dk = du * k_scale
            grr, gzz, gsh = _dfa_green_integrand(k, omega_c, h_m, vp_ms, vs_ms, rho)
            w = k * dk
            int_rr += max(-grr.imag, 0.0) * w
            int_zz += max(-gzz.imag, 0.0) * w
            int_sh += max(-gsh.imag, 0.0) * w

        ir = max(int_rr, 1e-30)
        iz = max(int_zz, 1e-30)
        ish = max(int_sh, 1e-30)
        h_rms_i = np.sqrt(0.5 * (ir + ish) / iz)
        hv_rms[fi] = h_rms_i
        hv_tot[fi] = h_rms_i * 1.4142135623730951
        hv_psv_rms[fi] = np.sqrt(0.5 * ir / iz)
        love_frac[fi] = ish / (ir + ish)

    return hv_rms, hv_tot, hv_psv_rms, love_frac


@njit(cache=False)
def _raydec_stack_chunk(
    zf: np.ndarray,
    nf: np.ndarray,
    ef: np.ndarray,
    zc_idx: np.ndarray,
    win_n: int,
    shift_q: int,
    align_n: int,
) -> Tuple[float, float, int]:
    """Hobiger et al. (2009) Eq. (2)-(3) Random Decrement stacking over positive-going Z
    zero-crossings with full-window quarter-period alignment and squared-correlation weights.
    """
    n_samp = len(zf)
    v_stack = np.zeros(win_n, dtype=np.float64)
    h_stack = np.zeros(win_n, dtype=np.float64)
    n_used = 0
    for k in range(len(zc_idx)):
        idx = zc_idx[k]
        if idx - shift_q < 0 or idx + win_n >= n_samp:
            continue
        cn = 0.0
        ce = 0.0
        ez = 0.0
        enn = 0.0
        eee = 0.0
        ene = 0.0
        for m in range(align_n):
            zv = zf[idx + m]
            nv = nf[idx - shift_q + m]
            ev = ef[idx - shift_q + m]
            cn += nv * zv
            ce += ev * zv
            ez += zv * zv
            enn += nv * nv
            eee += ev * ev
            ene += nv * ev
        norm_c = np.sqrt(cn * cn + ce * ce)
        if norm_c < 1e-30 or ez < 1e-30:
            continue
        cos_th = cn / norm_c
        sin_th = ce / norm_c
        eh_a = enn * cos_th * cos_th + eee * sin_th * sin_th + 2.0 * ene * cos_th * sin_th
        corr_i = norm_c / np.sqrt(ez * max(eh_a, 1e-30))
        w = corr_i * corr_i
        for m in range(win_n):
            v_stack[m] += w * zf[idx + m]
            h_stack[m] += w * (nf[idx - shift_q + m] * cos_th + ef[idx - shift_q + m] * sin_th)
        n_used += 1

    if n_used < 5:
        return np.nan, 0.0, n_used

    ev = 0.0
    eh = 0.0
    vh = 0.0
    for m in range(win_n):
        ev += v_stack[m] * v_stack[m]
        eh += h_stack[m] * h_stack[m]
        vh += v_stack[m] * h_stack[m]
    if ev <= 1e-30:
        return np.nan, 0.0, n_used
    ellip = np.sqrt(eh / ev)
    corr = vh / (np.sqrt(ev * eh) + 1e-30)
    return ellip, corr, n_used


def compute_raydec_ellipticity(
    e: np.ndarray,
    n: np.ndarray,
    z: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    chunk_sec: float = 600.0,
    dfpar: float = 0.10,
    cycles: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Apply Hobiger et al. (2009) RayDec across non-overlapping chunks of a 3C recording
    to extract pure Rayleigh-wave ellipticity (u_R / u_Z) free of Love/body waves.
    """
    n_samp = len(z)
    chunk_n = max(1, int(round(chunk_sec * fs)))
    n_chunks = max(4, n_samp // chunk_n)
    chunk_n = n_samp // n_chunks

    n_f = len(freqs)
    ellip_chunks = np.full((n_chunks, n_f), np.nan, dtype=np.float64)
    corr_chunks = np.full((n_chunks, n_f), np.nan, dtype=np.float64)
    n_zc_tot = np.zeros(n_f, dtype=np.int32)

    for fi, fc in enumerate(freqs):
        f_lo = max(0.12, fc * (1.0 - dfpar))
        f_hi = min(0.45 * fs, fc * (1.0 + dfpar))
        sos = signal.butter(3, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")

        zf = signal.sosfiltfilt(sos, z)
        nf = signal.sosfiltfilt(sos, n)
        ef = signal.sosfiltfilt(sos, e)

        period_samp = fs / fc
        win_n = max(16, int(round(cycles * period_samp)))
        shift_q = max(1, int(round(0.25 * period_samp)))
        align_n = win_n

        for ci in range(n_chunks):
            c0 = ci * chunk_n
            c1 = c0 + chunk_n
            zc_sub = zf[c0:c1]
            zc_idx = np.where((zc_sub[:-1] <= 0.0) & (zc_sub[1:] > 0.0))[0]
            if len(zc_idx) < 5:
                continue
            el, co, nu = _raydec_stack_chunk(
                zc_sub, nf[c0:c1], ef[c0:c1], zc_idx.astype(np.int64), win_n, shift_q, align_n
            )
            if np.isfinite(el) and el > 1e-4:
                ellip_chunks[ci, fi] = el
                corr_chunks[ci, fi] = co
                n_zc_tot[fi] += nu

    with np.errstate(all="ignore"):
        log_el = np.log10(ellip_chunks)
        log_mean = np.nanmean(log_el, axis=0)
        log_std = np.nanstd(log_el, axis=0, ddof=1)
        el_med = 10.0**log_mean
        el_p84 = 10.0 ** (log_mean + log_std)
        el_p16 = 10.0 ** (log_mean - log_std)
        corr_mean = np.nanmean(corr_chunks, axis=0)

    band = (freqs >= 0.25) & (freqs <= 10.0) & np.isfinite(el_med)
    fb = freqs[band]
    eb = el_med[band]
    ipk = int(np.argmax(eb)) if len(eb) > 0 else 0
    f0_rd = float(fb[ipk]) if len(fb) > 0 else float("nan")
    e0_rd = float(eb[ipk]) if len(eb) > 0 else float("nan")

    return {
        "freqs": freqs,
        "ellip_raydec": el_med,
        "ellip_p16": el_p16,
        "ellip_p84": el_p84,
        "ellip_log10_std": log_std,
        "ellip_rms_equiv": el_med / np.sqrt(2.0),
        "phase_lock_corr": corr_mean,
        "n_zero_crossings": n_zc_tot,
        "n_chunks": n_chunks,
        "f0_raydec_hz": f0_rd,
        "e0_raydec": e0_rd,
    }
