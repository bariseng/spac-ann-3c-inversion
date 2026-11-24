"""1D elastodynamic physics utilities, Brocher relations, SH transfer function,
engineering bedrock metrics (Z760_first / Z760_perm / Z1000 / Vs30), and synthetic
sedimentary-basin forward dataset generation using disba.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Dict, Tuple

import numpy as np
from disba import Ellipticity, PhaseDispersion

F_GRID_DEFAULT = np.geomspace(0.25, 15.0, 64)
Z_GRID_DEFAULT = np.array(
    [
        0, 2.5, 5, 7.5, 10, 12.5, 15, 18, 21, 25, 30, 35, 40, 45, 50,
        56, 62, 70, 78, 86, 95, 105, 115, 125, 138, 150, 165, 180, 195, 210,
        228, 246, 265, 285, 305, 325, 350, 375, 400, 425, 450, 480, 510, 540, 575,
        610, 650, 700,
    ],
    dtype=np.float64,
)


def brocher_vp_rho(
    h_m: np.ndarray, vs_ms: np.ndarray, z_wt: float = 6.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute realistic Vp (m/s) and density rho (g/cm^3) from Vs using Brocher (2005)
    with a saturated shallow water-table floor below z_wt.
    """
    n = len(vs_ms)
    tops = np.zeros(n, dtype=np.float64)
    if n > 1:
        tops[1:] = np.cumsum(h_m[:-1])
    vp = np.zeros_like(vs_ms, dtype=np.float64)
    rho = np.zeros_like(vs_ms, dtype=np.float64)
    for k in range(n):
        vsk = vs_ms[k] / 1000.0
        vp_b = (0.9409 + 2.0947 * vsk - 0.8206 * vsk**2 + 0.2683 * vsk**3 - 0.0251 * vsk**4) * 1000.0
        z_mid = tops[k] + (0.5 * h_m[k] if k < n - 1 else 50.0)
        if z_mid >= z_wt:
            vp_k = max(vp_b, 1550.0)
        else:
            vp_k = max(vs_ms[k] * 1.9, min(vp_b, 1450.0))
        vp[k] = max(vp_k, vs_ms[k] * 1.50)
        vpk = vp[k] / 1000.0
        rho_b = 1.6612 * vpk - 0.4721 * vpk**2 + 0.0671 * vpk**3 - 0.0043 * vpk**4 + 0.000106 * vpk**5
        rho[k] = float(np.clip(rho_b, 1.65, 2.60))
    return vp, rho


def sh_transfer_f0(
    h_m: np.ndarray, vs_ms: np.ndarray, rho_gcc: np.ndarray, freqs: np.ndarray
) -> Tuple[float, float]:
    """1D viscoelastic SH outcrop-to-surface transfer function via Haskell propagator.
    Returns fundamental peak frequency f0_sh (Hz) and peak amplification A0_sh.
    """
    n_lay = len(h_m) - 1
    xi = np.where(vs_ms < 500.0, 0.025, np.where(vs_ms < 900.0, 0.015, 0.008))
    vs_c = vs_ms * (1.0 + 1j * xi)
    mu = (rho_gcc * 1000.0) * (vs_c**2)

    omega = 2.0 * np.pi * freqs
    u = np.ones_like(omega, dtype=np.complex128)
    tau = np.zeros_like(omega, dtype=np.complex128)
    for k in range(n_lay):
        kk = omega / vs_c[k]
        kh = kk * h_m[k]
        c = np.cos(kh)
        s = np.sin(kh)
        zk = mu[k] * kk
        u, tau = c * u + (s / zk) * tau, (-zk * s) * u + c * tau

    z_hs = mu[-1] * (omega / vs_c[-1])
    tf = np.abs(1.0 / (u + tau / (1j * z_hs)))

    mask = (freqs >= 0.20) & (freqs <= 12.0)
    f_sub = freqs[mask]
    tf_sub = tf[mask]
    is_pk = np.zeros(len(tf_sub), dtype=bool)
    is_pk[1:-1] = (tf_sub[1:-1] > tf_sub[:-2]) & (tf_sub[1:-1] > tf_sub[2:]) & (tf_sub[1:-1] > 1.35)
    pks = np.where(is_pk)[0]
    if len(pks) == 0:
        idx = int(np.argmax(tf_sub))
        return float(f_sub[idx]), float(tf_sub[idx])
    gmax = float(np.max(tf_sub[pks]))
    sig_pks = [p for p in pks if tf_sub[p] >= max(1.5, 0.50 * gmax)]
    p0 = sig_pks[0] if sig_pks else pks[int(np.argmax(tf_sub[pks]))]
    return float(f_sub[p0]), float(tf_sub[p0])


def compute_profile_metrics(
    h_m: np.ndarray,
    vs_ms: np.ndarray,
    rho_gcc: np.ndarray,
    z_grid: np.ndarray = Z_GRID_DEFAULT,
) -> Dict[str, object]:
    """Compute exact Z760_first, Z760_perm, Z1000_perm, main impedance contrast depth,
    time-averaged Vs30/Vs100/Vs300, and sampled Vs(z_grid).
    """
    n_tot = len(vs_ms)
    tops = np.zeros(n_tot, dtype=np.float64)
    if n_tot > 1:
        tops[1:] = np.cumsum(h_m[:-1])

    idx_ge760 = np.where(vs_ms >= 760.0)[0]
    z760_first = float(tops[idx_ge760[0]]) if len(idx_ge760) > 0 else 750.0

    if vs_ms[-1] >= 760.0:
        k = n_tot - 1
        while k > 0 and vs_ms[k - 1] >= 760.0:
            k -= 1
        z760_perm = float(tops[k])
        has_z760 = 1.0 if z760_perm <= 700.0 else 0.0
    else:
        z760_perm = 750.0
        has_z760 = 0.0

    if vs_ms[-1] >= 1000.0:
        k = n_tot - 1
        while k > 0 and vs_ms[k - 1] >= 1000.0:
            k -= 1
        z1000_perm = float(tops[k])
        has_z1000 = 1.0 if z1000_perm <= 700.0 else 0.0
    else:
        z1000_perm = 750.0
        has_z1000 = 0.0

    imp = rho_gcc * vs_ms
    ratios = imp[1:] / np.maximum(imp[:-1], 1e-6)
    z_imp_max = float(tops[1 + int(np.argmax(ratios))]) if n_tot > 1 else 0.0

    def avg_vs(depth_m: float) -> float:
        rem = depth_m
        tt = 0.0
        for i in range(n_tot):
            thick = h_m[i] if i < n_tot - 1 else rem
            used = min(rem, thick)
            tt += used / vs_ms[i]
            rem -= used
            if rem <= 1e-9:
                break
        return float(depth_m / tt)

    idx_z = np.clip(np.searchsorted(tops, z_grid, side="right") - 1, 0, n_tot - 1)
    vs_grid = vs_ms[idx_z].astype(np.float32)

    return {
        "vs_grid": vs_grid,
        "z760_first": z760_first,
        "z760_perm": z760_perm,
        "has_z760": has_z760,
        "z1000_perm": z1000_perm,
        "has_z1000": has_z1000,
        "z_imp_max": z_imp_max,
        "vs30": avg_vs(30.0),
        "vs100": avg_vs(100.0),
        "vs300": avg_vs(300.0),
    }


def sample_basin_model(
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one physically realistic 1D sedimentary-basin / margin earth model
    (thickness_m, vp_ms, vs_ms, rho_gcc) spanning 0-750 m, including velocity inversions
    and ~8% ultra-deep soft profiles without Z760 bedrock.
    """
    regime_u = rng.random()
    if regime_u < 0.18:
        target_z760 = float(np.exp(rng.uniform(np.log(12.0), np.log(65.0))))
        has_bedrock = True
    elif regime_u < 0.43:
        target_z760 = float(np.exp(rng.uniform(np.log(65.0), np.log(180.0))))
        has_bedrock = True
    elif regime_u < 0.70:
        target_z760 = float(np.exp(rng.uniform(np.log(180.0), np.log(380.0))))
        has_bedrock = True
    elif regime_u < 0.92:
        target_z760 = float(np.exp(rng.uniform(np.log(380.0), np.log(630.0))))
        has_bedrock = True
    else:
        target_z760 = float(rng.uniform(680.0, 800.0))
        has_bedrock = False

    n_layers = int(rng.integers(4, 9))
    if has_bedrock:
        k_bed = int(rng.integers(max(1, n_layers - 3), n_layers + 1))
        raw_w = rng.dirichlet(np.linspace(0.7, 2.2, k_bed))
        h_above = np.maximum(raw_w * target_z760, 2.5)
        h_above = h_above * (target_z760 / np.sum(h_above))

        n_below = n_layers - k_bed
        if n_below > 0:
            rem_depth = max(60.0, 740.0 - target_z760)
            h_below = np.maximum(
                rng.dirichlet(np.ones(n_below) * 1.5) * rng.uniform(0.35, 0.95) * rem_depth, 15.0
            )
            h_m = np.concatenate([h_above, h_below, [0.0]])
        else:
            h_m = np.concatenate([h_above, [0.0]])

        vs = np.zeros(n_layers + 1, dtype=np.float64)
        vs[0] = rng.uniform(100.0, 320.0) if target_z760 > 120.0 else rng.uniform(130.0, 450.0)
        for k in range(1, k_bed):
            frac = k / float(k_bed)
            vs_base = vs[0] + frac * (rng.uniform(420.0, 720.0) - vs[0])
            vs[k] = np.clip(vs_base * rng.uniform(0.85, 1.18), 110.0, 740.0)

        vs[k_bed] = rng.uniform(765.0, 1450.0) if rng.random() < 0.75 else rng.uniform(1100.0, 2100.0)
        for k in range(k_bed + 1, n_layers + 1):
            vs[k] = np.clip(vs[k - 1] * rng.uniform(1.05, 1.55), vs[k - 1] + 20.0, 2500.0)

        inv_u = rng.random()
        if inv_u < 0.34 and k_bed >= 2:
            if inv_u < 0.14 and k_bed >= 3:
                k_stiff = int(rng.integers(1, k_bed - 1))
                vs[k_stiff] = rng.uniform(770.0, 1120.0)
                vs[k_stiff + 1] = rng.uniform(320.0, 680.0)
            else:
                k_lvz = int(rng.integers(1, k_bed))
                vs[k_lvz] = np.clip(vs[k_lvz - 1] * rng.uniform(0.55, 0.85), 110.0, 680.0)
    else:
        total_h = rng.uniform(450.0, 740.0)
        h_fin = np.maximum(rng.dirichlet(np.linspace(0.6, 2.0, n_layers)) * total_h, 5.0)
        h_m = np.concatenate([h_fin, [0.0]])
        vs = np.zeros(n_layers + 1, dtype=np.float64)
        vs[0] = rng.uniform(100.0, 280.0)
        vs_top_hs = rng.uniform(420.0, 740.0)
        for k in range(1, n_layers + 1):
            frac = k / float(n_layers)
            vs[k] = np.clip(vs[0] + frac * (vs_top_hs - vs[0]) * rng.uniform(0.85, 1.12), 110.0, 745.0)
        if rng.random() < 0.30 and n_layers >= 3:
            k_lvz = int(rng.integers(1, n_layers))
            vs[k_lvz] = np.clip(vs[k_lvz - 1] * rng.uniform(0.60, 0.85), 110.0, 650.0)

    tops = np.zeros(len(vs))
    tops[1:] = np.cumsum(h_m[:-1])
    z_wt = rng.uniform(1.0, 22.0)

    vp = np.zeros_like(vs)
    rho = np.zeros_like(vs)
    for k in range(len(vs)):
        vsk = vs[k] / 1000.0
        vp_b = (0.9409 + 2.0947 * vsk - 0.8206 * vsk**2 + 0.2683 * vsk**3 - 0.0251 * vsk**4) * 1000.0
        if tops[k] + (0.5 * h_m[k] if k < len(vs) - 1 else 50.0) >= z_wt:
            vp_k = max(vp_b, rng.uniform(1480.0, 1680.0))
        else:
            nu = rng.uniform(0.25, 0.38)
            vp_k = vs[k] * np.sqrt(2.0 * (1.0 - nu) / (1.0 - 2.0 * nu))
        vp[k] = max(vp_k, vs[k] * 1.48)
        vpk = vp[k] / 1000.0
        rho_b = 1.6612 * vpk - 0.4721 * vpk**2 + 0.0671 * vpk**3 - 0.0043 * vpk**4 + 0.000106 * vpk**5
        rho[k] = float(np.clip(rho_b + rng.normal(0.0, 0.04), 1.60, 2.65))

    return h_m, vp, vs, rho


def _worker_batch(args: Tuple[int, int, np.ndarray, np.ndarray]) -> dict:
    n_target, seed, f_grid, z_grid = args
    rng = np.random.default_rng(seed)
    t_asc = (1.0 / f_grid)[::-1].copy()
    n_f, n_z = f_grid.size, z_grid.size

    c_r0 = np.full((n_target, n_f), np.nan, dtype=np.float32)
    c_r1 = np.full((n_target, n_f), np.nan, dtype=np.float32)
    c_l0 = np.full((n_target, n_f), np.nan, dtype=np.float32)
    ellip = np.full((n_target, n_f), np.nan, dtype=np.float32)
    f0_sh = np.zeros(n_target, dtype=np.float32)
    a0_sh = np.zeros(n_target, dtype=np.float32)
    vs_grid = np.zeros((n_target, n_z), dtype=np.float32)
    z760_first = np.zeros(n_target, dtype=np.float32)
    z760_perm = np.zeros(n_target, dtype=np.float32)
    has_z760 = np.zeros(n_target, dtype=np.float32)
    z1000_perm = np.zeros(n_target, dtype=np.float32)
    has_z1000 = np.zeros(n_target, dtype=np.float32)

    f_sh_grid = np.geomspace(0.20, 15.0, 128)
    valid_count, attempts = 0, 0

    def map_f(periods, values):
        arr = np.full(n_f, np.nan, dtype=np.float32)
        if len(periods) == 0:
            return arr
        ix = np.clip(np.searchsorted(t_asc, periods), 0, n_f - 1)
        ok = np.abs(t_asc[ix] - periods) < (1e-4 * periods + 1e-6)
        arr[(n_f - 1) - ix[ok]] = values[ok]
        return arr

    while valid_count < n_target and attempts < n_target * 2:
        attempts += 1
        h_m, vp_ms, vs_ms, rho_gcc = sample_basin_model(rng)
        vm = np.column_stack([h_m / 1000.0, vp_ms / 1000.0, vs_ms / 1000.0, rho_gcc])
        try:
            pd_calc = PhaseDispersion(*vm.T)
            r0 = pd_calc(t_asc, mode=0, wave="rayleigh")
            if len(r0.velocity) < n_f * 0.85:
                continue
            v0 = map_f(r0.period, r0.velocity * 1000.0)
            if np.isnan(v0).sum() > 6:
                continue
            if np.isnan(v0).any():
                g = ~np.isnan(v0)
                v0 = np.interp(np.log(f_grid), np.log(f_grid[g]), v0[g]).astype(np.float32)

            try:
                r1 = pd_calc(t_asc, mode=1, wave="rayleigh")
                v1 = map_f(r1.period, r1.velocity * 1000.0)
            except Exception:
                v1 = np.full(n_f, np.nan, dtype=np.float32)
            try:
                l0 = pd_calc(t_asc, mode=0, wave="love")
                vl0 = map_f(l0.period, l0.velocity * 1000.0)
                if np.isnan(vl0).any() and np.isfinite(vl0).sum() >= 6:
                    gl = np.isfinite(vl0)
                    vl0 = np.interp(np.log(f_grid), np.log(f_grid[gl]), vl0[gl]).astype(np.float32)
            except Exception:
                vl0 = np.full(n_f, np.nan, dtype=np.float32)
            try:
                el0 = Ellipticity(*vm.T)(t_asc, mode=0)
                e_arr = map_f(el0.period, np.abs(el0.ellipticity))
            except Exception:
                e_arr = np.full(n_f, np.nan, dtype=np.float32)

            f0s, a0s = sh_transfer_f0(h_m, vs_ms, rho_gcc, f_sh_grid)
            met = compute_profile_metrics(h_m, vs_ms, rho_gcc, z_grid=z_grid)

            c_r0[valid_count] = v0
            c_r1[valid_count] = v1
            c_l0[valid_count] = vl0
            ellip[valid_count] = e_arr
            f0_sh[valid_count] = f0s
            a0_sh[valid_count] = a0s
            vs_grid[valid_count] = met["vs_grid"]
            z760_first[valid_count] = met["z760_first"]
            z760_perm[valid_count] = met["z760_perm"]
            has_z760[valid_count] = met["has_z760"]
            z1000_perm[valid_count] = met["z1000_perm"]
            has_z1000[valid_count] = met["has_z1000"]
            valid_count += 1
        except Exception:
            continue

    return {
        "c_r0": c_r0[:valid_count],
        "c_r1": c_r1[:valid_count],
        "c_l0": c_l0[:valid_count],
        "ellip": ellip[:valid_count],
        "f0_sh": f0_sh[:valid_count],
        "a0_sh": a0_sh[:valid_count],
        "vs_grid": vs_grid[:valid_count],
        "z760_first": z760_first[:valid_count],
        "z760_perm": z760_perm[:valid_count],
        "has_z760": has_z760[:valid_count],
        "z1000_perm": z1000_perm[:valid_count],
        "has_z1000": has_z1000[:valid_count],
    }


def generate_synthetic_dataset(
    n_models: int = 10000,
    n_workers: int = 4,
    out_path: str | None = None,
    seed: int = 42,
    f_grid: np.ndarray = F_GRID_DEFAULT,
    z_grid: np.ndarray = Z_GRID_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic sedimentary-basin dataset in parallel using disba."""
    n_workers = max(1, min(n_workers, os.cpu_count() or 1))
    per_w = int(np.ceil(n_models / n_workers))
    tasks = [(per_w, seed + i * 97, f_grid, z_grid) for i in range(n_workers)]
    if n_workers == 1:
        parts = [_worker_batch(tasks[0])]
    else:
        with mp.Pool(processes=n_workers) as pool:
            parts = pool.map(_worker_batch, tasks)

    merged: Dict[str, np.ndarray] = {}
    for k in parts[0].keys():
        merged[k] = np.concatenate([p[k] for p in parts], axis=0)[:n_models]
    merged["f_grid"] = f_grid.astype(np.float32)
    merged["z_grid"] = z_grid.astype(np.float32)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez_compressed(out_path, **merged)
    return merged
