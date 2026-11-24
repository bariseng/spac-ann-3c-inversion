"""Trans-Dimensional Reversible-Jump MCMC (Green 1995 / Bodin et al. 2012 / BayesBay)
with 4-temperature Parallel Tempering (Replica Exchange) over joint Rayleigh R0,
Love L0, and 1D SH resonance frequency f0 using disba.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import numpy as np
from disba import PhaseDispersion

from .physics import Z_GRID_DEFAULT, brocher_vp_rho


def _sh_f0_fast(h_m: np.ndarray, vs_ms: np.ndarray, rho_gcc: np.ndarray, freqs: np.ndarray) -> float:
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
    is_pk = np.zeros(len(tf), dtype=bool)
    is_pk[1:-1] = (tf[1:-1] > tf[:-2]) & (tf[1:-1] > tf[2:]) & (tf[1:-1] > 1.35)
    pks = np.where(is_pk)[0]
    if len(pks) == 0:
        return float(freqs[int(np.argmax(tf))])
    gmax = float(np.max(tf[pks]))
    for p in pks:
        if tf[p] >= max(1.5, 0.50 * gmax):
            return float(freqs[p])
    return float(freqs[pks[int(np.argmax(tf[pks]))]])


def _rjmcmc_log_likelihood_split(
    z_if: np.ndarray,
    vs: np.ndarray,
    t_r_asc: np.ndarray,
    c_r_obs_rev: np.ndarray,
    s_r_rev: np.ndarray,
    t_l_asc: np.ndarray,
    c_l_obs_rev: np.ndarray,
    s_l_rev: np.ndarray,
    f0_obs: float,
    hvsr_rel: bool,
    f_sh_grid: np.ndarray,
    z_grid: np.ndarray,
    d_max_m: float,
    log_vs_deep_prior: np.ndarray,
) -> Tuple[float, float]:
    """Return (ll_data, ll_phys) separately for exact Parallel Tempering Replica Exchange."""
    h_m = np.empty(len(vs), dtype=np.float64)
    if len(z_if) > 0:
        h_m[0] = z_if[0]
        h_m[1:-1] = np.diff(z_if)
        if np.any(h_m[:-1] < 3.0):
            return -1e12, -1e12
    h_m[-1] = 0.0
    if np.any(vs < 85.0) or np.any(vs > 2500.0):
        return -1e12, -1e12

    vp_ms, rho_gcc = brocher_vp_rho(h_m, vs)
    vm = np.column_stack([h_m / 1000.0, vp_ms / 1000.0, vs / 1000.0, rho_gcc])
    try:
        pd_c = PhaseDispersion(*vm.T)
        r0 = pd_c(t_r_asc, mode=0, wave="rayleigh")
        if len(r0.velocity) < len(t_r_asc):
            return -1e12, -1e12
        cr_mod = r0.velocity * 1000.0
        ll_r = -0.5 * float(np.sum(((c_r_obs_rev - cr_mod) / s_r_rev) ** 2))

        ll_l = 0.0
        if len(t_l_asc) >= 4:
            l0 = pd_c(t_l_asc, mode=0, wave="love")
            if len(l0.velocity) < len(t_l_asc):
                return -1e12, -1e12
            cl_mod = l0.velocity * 1000.0
            ll_l = -0.5 * float(np.sum(((c_l_obs_rev - cl_mod) / (1.25 * s_l_rev)) ** 2))
    except Exception:
        return -1e12, -1e12

    ll_f0 = 0.0
    if hvsr_rel and np.isfinite(f0_obs) and f0_obs > 0.15:
        f0_m = _sh_f0_fast(h_m, vs, rho_gcc, f_sh_grid)
        ll_f0 = -0.5 * ((math.log(max(f0_m, 0.1)) - math.log(max(f0_obs, 0.1))) / 0.09) ** 2 * 4.0

    dv = np.diff(np.log(vs))
    ll_phys = -2.5 * float(np.sum(np.maximum(0.0, -dv - 0.55) ** 2))
    deep_m = z_grid > d_max_m
    if np.any(deep_m):
        tops = np.concatenate([[0.0], z_if])
        iz = np.clip(np.searchsorted(tops, z_grid[deep_m], side="right") - 1, 0, len(vs) - 1)
        ll_phys -= 0.5 * float(np.mean(((np.log(vs[iz]) - log_vs_deep_prior[deep_m]) / 0.35) ** 2)) * 2.0

    return (ll_r + ll_l + ll_f0), ll_phys


def _rjmcmc_propose_move(
    z_if: np.ndarray,
    vs: np.ndarray,
    rng: np.random.Generator,
    k_min: int = 3,
    k_max: int = 10,
    log_z_lo: float = math.log(4.0),
    log_z_hi: float = math.log(620.0),
    log_v_lo: float = math.log(90.0),
    log_v_hi: float = math.log(2450.0),
    sigma_v: float = 0.075,
    sigma_z: float = 0.10,
    sigma_b: float = 0.25,
):
    """Exact detailed-balance trans-dimensional rj-MCMC proposal (Green 1995 / Bodin 2012).
    Guarantees p_birth(k) == p_death(k+1) == 0.14 identically for all k in [k_min, k_max-1].
    """
    k = len(vs)
    u_move = rng.random()
    log_q_ratio = 0.0
    prop_z, prop_vs = z_if, vs

    if u_move < 0.50:
        j = int(rng.integers(0, k))
        prop_vs = vs.copy()
        lv_new = math.log(vs[j]) + rng.normal(0.0, sigma_v)
        if log_v_lo <= lv_new <= log_v_hi:
            prop_vs[j] = math.exp(lv_new)
        else:
            prop_vs = None
    elif u_move < 0.72:
        j = int(rng.integers(0, k - 1))
        prop_z = z_if.copy()
        lz_new = math.log(z_if[j]) + rng.normal(0.0, sigma_z)
        if log_z_lo <= lz_new <= log_z_hi:
            z_new = math.exp(lz_new)
            lo_b = (z_if[j - 1] + 3.0) if j > 0 else 4.0
            hi_b = (z_if[j + 1] - 3.0) if j < k - 2 else 620.0
            if lo_b < z_new < hi_b:
                prop_z[j] = z_new
            else:
                prop_z = None
        else:
            prop_z = None
    elif u_move < 0.86:
        if k < k_max:
            z_new = math.exp(rng.uniform(log_z_lo, log_z_hi))
            pos = int(np.searchsorted(z_if, z_new))
            ok_lo = (z_new - z_if[pos - 1] >= 3.0) if pos > 0 else (z_new >= 4.0)
            ok_hi = (z_if[pos] - z_new >= 3.0) if pos < len(z_if) else (z_new <= 620.0)
            if ok_lo and ok_hi:
                lv_parent = math.log(vs[pos])
                lv_new = lv_parent + rng.normal(0.0, sigma_b)
                if log_v_lo <= lv_new <= log_v_hi:
                    prop_z = np.insert(z_if, pos, z_new)
                    prop_vs = np.insert(vs, pos, math.exp(lv_new))
                    log_prior_v = -math.log(log_v_hi - log_v_lo)
                    log_prop_v = (
                        -0.5 * math.log(2.0 * math.pi)
                        - math.log(sigma_b)
                        - 0.5 * ((lv_new - lv_parent) / sigma_b) ** 2
                    )
                    log_q_ratio = log_prior_v - log_prop_v
                else:
                    prop_z = None
            else:
                prop_z = None
        else:
            prop_z = None
    else:
        if k > k_min:
            pos = int(rng.integers(0, k - 1))
            lv_del = math.log(vs[pos])
            lv_adj = math.log(vs[pos + 1])
            prop_z = np.delete(z_if, pos)
            prop_vs = np.delete(vs, pos)
            log_prior_v = -math.log(log_v_hi - log_v_lo)
            log_prop_v = (
                -0.5 * math.log(2.0 * math.pi)
                - math.log(sigma_b)
                - 0.5 * ((lv_del - lv_adj) / sigma_b) ** 2
            )
            log_q_ratio = log_prop_v - log_prior_v
        else:
            prop_z = None

    return prop_z, prop_vs, log_q_ratio


def _rjmcmc_chain_worker(args):
    (
        chain_id,
        seed,
        n_iter,
        burn_in,
        thin,
        h_init,
        vs_init,
        t_r_asc,
        c_r_rev,
        s_r_rev,
        t_l_asc,
        c_l_rev,
        s_l_rev,
        f0_obs,
        hvsr_rel,
        z_grid,
        d_max_m,
        log_vs_deep_prior,
    ) = args

    rng = np.random.default_rng(seed)
    k_min, k_max = 3, 10
    f_sh_grid = np.geomspace(0.20, 12.0, 96)

    temps = (1.0, 2.2, 5.5, 14.0)
    betas = [1.0 / T for T in temps]
    n_t = len(temps)
    step_scales = [1.0, 1.35, 1.85, 2.40]

    rep_z, rep_vs, rep_lld, rep_llp = [], [], [], []
    for ti in range(n_t):
        z_if = np.cumsum(h_init[:-1]).astype(np.float64)
        vs = vs_init.astype(np.float64).copy()
        z_if = np.sort(np.clip(z_if * np.exp(rng.normal(0.0, 0.16 * step_scales[ti], size=len(z_if))), 4.5, 600.0))
        for j in range(1, len(z_if)):
            if z_if[j] < z_if[j - 1] + 3.5:
                z_if[j] = z_if[j - 1] + 3.5
        z_if = z_if[z_if < 615.0]
        vs = np.clip(
            vs[: len(z_if) + 1] * np.exp(rng.normal(0.0, 0.12 * step_scales[ti], size=len(z_if) + 1)),
            95.0,
            2400.0,
        )
        lld, llp = _rjmcmc_log_likelihood_split(
            z_if, vs, t_r_asc, c_r_rev, s_r_rev, t_l_asc, c_l_rev, s_l_rev,
            f0_obs, hvsr_rel, f_sh_grid, z_grid, d_max_m, log_vs_deep_prior,
        )
        if lld < -1e11:
            z_if = np.cumsum(h_init[:-1]).astype(np.float64)
            vs = vs_init.astype(np.float64).copy()
            lld, llp = _rjmcmc_log_likelihood_split(
                z_if, vs, t_r_asc, c_r_rev, s_r_rev, t_l_asc, c_l_rev, s_l_rev,
                f0_obs, hvsr_rel, f_sh_grid, z_grid, d_max_m, log_vs_deep_prior,
            )
        rep_z.append(z_if)
        rep_vs.append(vs)
        rep_lld.append(lld)
        rep_llp.append(llp)

    n_keep = max(1, (n_iter - burn_in) // thin)
    k_samples = np.zeros(n_keep, dtype=np.int32)
    vs_samples = np.zeros((n_keep, len(z_grid)), dtype=np.float32)
    z760f_samples = np.zeros(n_keep, dtype=np.float32)
    z760p_samples = np.zeros(n_keep, dtype=np.float32)
    vs30_samples = np.zeros(n_keep, dtype=np.float32)
    vs100_samples = np.zeros(n_keep, dtype=np.float32)

    n_acc_cold, n_swap_try, n_swap_acc, keep_i = 0, 0, 0, 0
    for it in range(n_iter):
        for ti in range(n_t):
            sc = step_scales[ti]
            prop_z, prop_vs, log_q_ratio = _rjmcmc_propose_move(
                rep_z[ti], rep_vs[ti], rng, k_min=k_min, k_max=k_max,
                sigma_v=0.075 * sc, sigma_z=0.10 * sc, sigma_b=0.25,
            )
            if prop_z is not None and prop_vs is not None:
                p_lld, p_llp = _rjmcmc_log_likelihood_split(
                    prop_z, prop_vs, t_r_asc, c_r_rev, s_r_rev, t_l_asc, c_l_rev, s_l_rev,
                    f0_obs, hvsr_rel, f_sh_grid, z_grid, d_max_m, log_vs_deep_prior,
                )
                if p_lld > -1e11:
                    cur_lp = betas[ti] * rep_lld[ti] + rep_llp[ti]
                    prop_lp = betas[ti] * p_lld + p_llp
                    log_alpha = (prop_lp - cur_lp) + log_q_ratio
                    if log_alpha >= 0.0 or math.log(rng.random()) < log_alpha:
                        rep_z[ti], rep_vs[ti], rep_lld[ti], rep_llp[ti] = prop_z, prop_vs, p_lld, p_llp
                        if ti == 0:
                            n_acc_cold += 1

        if it % 4 == 0:
            m = int(rng.integers(0, n_t - 1))
            n_swap_try += 1
            log_a_swap = (betas[m] - betas[m + 1]) * (rep_lld[m + 1] - rep_lld[m])
            if log_a_swap >= 0.0 or math.log(rng.random()) < log_a_swap:
                rep_z[m], rep_z[m + 1] = rep_z[m + 1], rep_z[m]
                rep_vs[m], rep_vs[m + 1] = rep_vs[m + 1], rep_vs[m]
                rep_lld[m], rep_lld[m + 1] = rep_lld[m + 1], rep_lld[m]
                rep_llp[m], rep_llp[m + 1] = rep_llp[m + 1], rep_llp[m]
                n_swap_acc += 1

        if it >= burn_in and ((it - burn_in) % thin == 0) and keep_i < n_keep:
            z_if = rep_z[0]
            vs = rep_vs[0]
            k_samples[keep_i] = len(vs)
            tops = np.concatenate([[0.0], z_if])
            iz = np.clip(np.searchsorted(tops, z_grid, side="right") - 1, 0, len(vs) - 1)
            vs_samples[keep_i] = vs[iz].astype(np.float32)

            idx760 = np.where(vs >= 760.0)[0]
            z760f_samples[keep_i] = float(tops[idx760[0]]) if len(idx760) > 0 else 750.0
            if vs[-1] >= 760.0:
                kk = len(vs) - 1
                while kk > 0 and vs[kk - 1] >= 760.0:
                    kk -= 1
                z760p_samples[keep_i] = float(tops[kk])
            else:
                z760p_samples[keep_i] = 750.0

            h_cur = np.empty(len(vs))
            h_cur[0] = z_if[0]
            h_cur[1:-1] = np.diff(z_if)
            h_cur[-1] = 1e6
            for D, arr_out in ((30.0, vs30_samples), (100.0, vs100_samples)):
                rem, tt = D, 0.0
                for j in range(len(vs)):
                    u = min(rem, h_cur[j])
                    tt += u / vs[j]
                    rem -= u
                    if rem <= 1e-9:
                        break
                arr_out[keep_i] = float(D / tt)
            keep_i += 1

    return {
        "chain_id": chain_id,
        "acc_rate": n_acc_cold / float(n_iter),
        "swap_rate": n_swap_acc / float(max(n_swap_try, 1)),
        "k": k_samples[:keep_i],
        "vs": vs_samples[:keep_i],
        "z760f": z760f_samples[:keep_i],
        "z760p": z760p_samples[:keep_i],
        "vs30": vs30_samples[:keep_i],
        "vs100": vs100_samples[:keep_i],
    }


def gelman_rubin_rhat(chains_1d: List[np.ndarray]) -> float:
    """Classic Gelman-Rubin potential scale reduction factor R-hat across M chains."""
    m = len(chains_1d)
    n = min(len(c) for c in chains_1d)
    if m < 2 or n < 10:
        return float("nan")
    arr = np.vstack([c[:n] for c in chains_1d])
    chain_means = np.mean(arr, axis=1)
    chain_vars = np.var(arr, axis=1, ddof=1)
    W = float(np.mean(chain_vars))
    B = float(n * np.var(chain_means, ddof=1))
    if W <= 1e-12:
        return 1.0
    var_hat = ((n - 1.0) / n) * W + (1.0 / n) * B
    return float(np.sqrt(max(var_hat / W, 1.0)))


def run_parallel_tempering_rjmcmc(
    h_init: np.ndarray,
    vs_init: np.ndarray,
    f_r0: np.ndarray,
    c_r0: np.ndarray,
    s_r0: np.ndarray,
    f_l0: np.ndarray,
    c_l0: np.ndarray,
    s_l0: np.ndarray,
    f0_obs: float,
    hvsr_reliable: bool,
    d_max_m: float,
    log_vs_deep_prior: np.ndarray,
    z_grid: np.ndarray = Z_GRID_DEFAULT,
    n_chains: int = 8,
    n_iter: int = 4500,
    burn_in: int = 1500,
    thin: int = 3,
    seed: int = 2026,
) -> Dict[str, object]:
    """Run multi-chain 4-temperature Parallel-Tempering rj-MCMC and summarize posteriors."""
    t_r_asc = (1.0 / f_r0)[::-1].copy()
    c_r_rev = c_r0[::-1].copy()
    s_r_rev = np.maximum(s_r0[::-1], 0.04 * c_r_rev)

    if len(f_l0) >= 4:
        t_l_asc = (1.0 / f_l0)[::-1].copy()
        c_l_rev = c_l0[::-1].copy()
        s_l_rev = np.maximum(s_l0[::-1], 0.05 * c_l_rev)
    else:
        t_l_asc = np.array([], dtype=np.float64)
        c_l_rev = np.array([], dtype=np.float64)
        s_l_rev = np.array([], dtype=np.float64)

    tasks = [
        (
            cid, seed + cid * 77, n_iter, burn_in, thin,
            h_init, vs_init, t_r_asc, c_r_rev, s_r_rev,
            t_l_asc, c_l_rev, s_l_rev, f0_obs, hvsr_reliable,
            z_grid, d_max_m, log_vs_deep_prior,
        )
        for cid in range(n_chains)
    ]
    n_workers = min(n_chains, os.cpu_count() or 1)
    if n_workers <= 1:
        res_chains = [_rjmcmc_chain_worker(t) for t in tasks]
    else:
        with mp.Pool(processes=n_workers) as pool:
            res_chains = pool.map(_rjmcmc_chain_worker, tasks)

    k_all = np.concatenate([c["k"] for c in res_chains])
    vs_all = np.vstack([c["vs"] for c in res_chains])
    z760f_all = np.concatenate([c["z760f"] for c in res_chains])
    z760p_all = np.concatenate([c["z760p"] for c in res_chains])
    vs30_all = np.concatenate([c["vs30"] for c in res_chains])
    vs100_all = np.concatenate([c["vs100"] for c in res_chains])

    return {
        "n_samples": len(z760p_all),
        "acc_rate_mean": float(np.mean([c["acc_rate"] for c in res_chains])),
        "swap_rate_mean": float(np.mean([c["swap_rate"] for c in res_chains])),
        "rhat_z760": gelman_rubin_rhat([c["z760p"] for c in res_chains]),
        "k_samples": k_all,
        "vs_samples": vs_all,
        "z760_first_samples": z760f_all,
        "z760_perm_samples": z760p_all,
        "vs30_samples": vs30_all,
        "vs100_samples": vs100_all,
    }
