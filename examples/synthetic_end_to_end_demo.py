"""Self-contained synthetic end-to-end demonstration of ``spac_ann``.

Runs cleanly without external data files by synthesizing a 5-layer sedimentary
basin over elastic bedrock, simulating decoupled 3C SPAC/F-J wavefields,
computing elastic DFA HVSR, running two-stage PyTorch ADsurf joint inversion,
verifying ANN/CNF posterior heads, and sampling via Parallel-Tempering rj-MCMC.

Figures are written to ``./demo_outputs/``.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from disba import Ellipticity, PhaseDispersion
from scipy import special

# Allow running directly from repository root without editable install
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spac_ann import (
    ConditionalNormalizingFlow3C,
    JointNetDispFormer,
    MDNInversionNet,
    Z_GRID_DEFAULT,
    brocher_vp_rho,
    compute_decoupled_3c_fj,
    compute_dfa_hvsr_full,
    compute_profile_metrics,
    invert_adsurf_joint,
    love_det_torch,
    newton_velocity_residual_torch,
    rayleigh_det_plucker_torch,
    run_parallel_tempering_rjmcmc,
    sh_transfer_f0,
)
from spac_ann.plotting import plot_dispersion_fit, plot_vs_uncertainty


def main() -> None:
    out_dir = os.path.join(os.getcwd(), "demo_outputs")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(2026)

    print("=== 1. Synthetic Basin Model & Exact Forward Simulation ===")
    h_true = np.array([15.0, 45.0, 110.0, 180.0, 0.0], dtype=np.float64)
    vs_true = np.array([190.0, 320.0, 520.0, 740.0, 1450.0], dtype=np.float64)
    vp_true, rho_true = brocher_vp_rho(h_true, vs_true, z_wt=6.0)
    met_true = compute_profile_metrics(h_true, vs_true, rho_true)
    f0_true, _ = sh_transfer_f0(h_true, vs_true, rho_true, np.geomspace(0.2, 12.0, 160))
    print(
        f"  True profile: Vs30={met_true['vs30']:.1f} m/s, Vs100={met_true['vs100']:.1f} m/s, "
        f"Z760={met_true['z760_perm']:.1f} m, f0_SH={f0_true:.3f} Hz"
    )

    f_disp = np.geomspace(0.30, 8.0, 36)
    t_asc = (1.0 / f_disp)[::-1].copy()
    vm = np.column_stack([h_true / 1000.0, vp_true / 1000.0, vs_true / 1000.0, rho_true])
    pd_c = PhaseDispersion(*vm.T)
    r0_res = pd_c(t_asc, mode=0, wave="rayleigh")
    l0_res = pd_c(t_asc, mode=0, wave="love")
    c_r0_true = (r0_res.velocity * 1000.0)[::-1].copy()
    c_l0_true = (l0_res.velocity * 1000.0)[::-1].copy()

    f_rd = np.geomspace(0.25, 10.0, 40)
    hv_dfa_true, _, _, _ = compute_dfa_hvsr_full(h_true, vp_true, vs_true, rho_true, f_rd, damping=0.012)

    el_raw = Ellipticity(*vm.T)((1.0 / f_rd)[::-1].copy(), mode=0)
    el_interp = np.interp(f_rd, 1.0 / el_raw.period[::-1], np.abs(el_raw.ellipticity[::-1]))

    print("\n=== 2. Decoupled 3C SPAC / F-J Wavefield Simulation & Recovery ===")
    pair_r_m = np.linspace(18.0, 450.0, 24)
    kr = 2.0 * np.pi * f_disp[None, :] * pair_r_m[:, None] / c_r0_true[None, :]
    kl = 2.0 * np.pi * f_disp[None, :] * pair_r_m[:, None] / c_l0_true[None, :]
    alpha_love = np.full_like(f_disp, 0.55)

    rho_zz = special.j0(kr) + rng.normal(0.0, 0.02, size=kr.shape)
    rho_tt = alpha_love[None, :] * (special.j0(kl) - special.jv(2, kl)) + (
        1.0 - alpha_love[None, :]
    ) * (special.j0(kr) + special.jv(2, kr)) + rng.normal(0.0, 0.02, size=kl.shape)
    rho_rr = alpha_love[None, :] * (special.j0(kl) + special.jv(2, kl)) + (
        1.0 - alpha_love[None, :]
    ) * (special.j0(kr) - special.jv(2, kr)) + rng.normal(0.0, 0.02, size=kl.shape)
    rho_zr_anti = 0.35 * special.j1(kr) + rng.normal(0.0, 0.015, size=kr.shape)

    c_grid = np.linspace(110.0, 1600.0, 160)
    fj_res = compute_decoupled_3c_fj(
        f_disp, c_grid, pair_r_m, rho_zz, rho_rr, rho_tt, rho_zr_anti=rho_zr_anti, c_r0_guide=c_r0_true
    )
    c_l0_obs = fj_res["c_l0_track"]
    c_r0_obs = c_r0_true * (1.0 + rng.normal(0.0, 0.025, size=len(f_disp)))
    l0_track_err = float(np.mean(np.abs(c_l0_obs - c_l0_true) / c_l0_true) * 100.0)
    print(f"  3C F-J decoupled Love L0 recovery MAPE: {l0_track_err:.2f}%")

    print("\n=== 3. PyTorch ADsurf Mode-Number-Free Joint Inversion ===")
    h_init = np.array([18.0, 55.0, 95.0, 150.0, 0.0], dtype=np.float64)
    vs_init = np.array([220.0, 380.0, 470.0, 650.0, 1250.0], dtype=np.float64)

    with torch.no_grad():
        ht = torch.tensor(h_true, dtype=torch.float64)
        vpt = torch.tensor(vp_true, dtype=torch.float64)
        vst = torch.tensor(vs_true, dtype=torch.float64)
        rhot = torch.tensor(rho_true, dtype=torch.float64)
        res_r = newton_velocity_residual_torch(
            rayleigh_det_plucker_torch,
            torch.tensor(f_disp, dtype=torch.float64),
            torch.tensor(c_r0_true, dtype=torch.float64),
            ht,
            vpt,
            vst,
            rhot,
        )
        res_l = newton_velocity_residual_torch(
            love_det_torch,
            torch.tensor(f_disp, dtype=torch.float64),
            torch.tensor(c_l0_true, dtype=torch.float64),
            ht,
            vst,
            rhot,
        )
        print(
            f"  True-model determinant residuals: Rayleigh mean={res_r.mean().item():.3e} m/s, "
            f"Love mean={res_l.mean().item():.3e} m/s"
        )

    inv_res = invert_adsurf_joint(
        h_init_base=h_init,
        vs_init_base=vs_init,
        f_disp=f_disp,
        c_r0_obs=c_r0_obs,
        c_r0_std=0.04 * c_r0_obs,
        c_l0_obs=c_l0_obs,
        c_l0_std=0.06 * c_l0_obs,
        alpha_love=alpha_love,
        pair_r_m=pair_r_m,
        rho_zz_corr=rho_zz,
        rho_tt_re=rho_tt,
        f_rd=f_rd,
        hvsr_obs_rd=hv_dfa_true,
        raydec_obs=el_interp,
        f0_obs=f0_true,
        hvsr_reliable=True,
        d_max_m=420.0,
        n_starts=2,
        n_steps_stage1=60,
        run_stage2=True,
    )
    vp_inv, rho_inv = brocher_vp_rho(inv_res["h_opt"], inv_res["vs_opt"], z_wt=6.0)
    met_inv = compute_profile_metrics(inv_res["h_opt"], inv_res["vs_opt"], rho_inv)
    f0_inv, _ = sh_transfer_f0(inv_res["h_opt"], inv_res["vs_opt"], rho_inv, np.geomspace(0.2, 12.0, 160))
    print(
        f"  ADsurf inverted: Vs30={met_inv['vs30']:.1f} m/s, Vs100={met_inv['vs100']:.1f} m/s, "
        f"Z760={met_inv['z760_perm']:.1f} m, f0_SH={f0_inv:.3f} Hz"
    )

    print("\n=== 4. Deep Learning Posterior Heads (MDN, JointNet, 3C CNF) ===")
    mdn = MDNInversionNet()
    jnet = JointNetDispFormer()
    cnf = ConditionalNormalizingFlow3C()
    d_in4 = torch.randn(2, 4, 64)
    h_in3 = torch.randn(2, 3, 64)
    s_in6 = torch.randn(2, 6)
    _ = mdn(d_in4, h_in3, s_in6)
    _ = jnet(d_in4, h_in3, s_in6)
    cnf_post = cnf.sample_posterior(torch.randn(1, 7, 64), torch.randn(1, 4, 64), torch.randn(1, 8), n_samples=128)
    print(f"  Sampled 3C CNF posterior shape: {cnf_post['vs_samples'].shape}")

    print("\n=== 5. Parallel-Tempering rj-MCMC Posterior Sampling ===")
    tops_inv = np.concatenate([[0.0], np.cumsum(inv_res["h_opt"][:-1])])
    iz_prior = np.clip(np.searchsorted(tops_inv, Z_GRID_DEFAULT, side="right") - 1, 0, len(inv_res["vs_opt"]) - 1)
    log_vs_prior = np.log(inv_res["vs_opt"][iz_prior])

    rj = run_parallel_tempering_rjmcmc(
        h_init=inv_res["h_opt"],
        vs_init=inv_res["vs_opt"],
        f_r0=f_disp,
        c_r0=c_r0_obs,
        s_r0=0.05 * c_r0_obs,
        f_l0=f_disp,
        c_l0=c_l0_obs,
        s_l0=0.07 * c_l0_obs,
        f0_obs=f0_true,
        hvsr_reliable=True,
        d_max_m=420.0,
        log_vs_deep_prior=log_vs_prior,
        n_chains=2,
        n_iter=600,
        burn_in=200,
        thin=2,
        seed=42,
    )
    z760_p16, z760_med, z760_p84 = np.percentile(rj["z760_perm_samples"], [16, 50, 84])
    print(
        f"  rj-MCMC samples={rj['n_samples']}, R-hat(Z760)={rj['rhat_z760']:.3f}, "
        f"Z760={z760_med:.1f} m [{z760_p16:.1f}, {z760_p84:.1f}]"
    )

    print("\n=== 6. Writing Verification Figures ===")
    z_fine = Z_GRID_DEFAULT[Z_GRID_DEFAULT <= 500.0]
    iz_z = np.searchsorted(Z_GRID_DEFAULT, z_fine)
    vs_profs = rj["vs_samples"][:, iz_z]
    vs_fig = os.path.join(out_dir, "demo_vs_uncertainty.png")
    disp_fig = os.path.join(out_dir, "demo_dispersion_fit.png")

    plot_vs_uncertainty(
        "Demo Basin",
        z_fine,
        vs_profs,
        inv_res["h_opt"],
        inv_res["vs_opt"],
        420.0,
        float(met_inv["vs30"]),
        float(met_inv["vs100"]),
        vs_fig,
    )

    vm_inv = np.column_stack([inv_res["h_opt"] / 1000.0, vp_inv / 1000.0, inv_res["vs_opt"] / 1000.0, rho_inv])
    r0_inv = PhaseDispersion(*vm_inv.T)(t_asc, mode=0, wave="rayleigh")
    c_best_dense = (r0_inv.velocity * 1000.0)[::-1].copy()
    acc_curves = [c_best_dense * (1.0 + rng.normal(0.0, 0.025, size=len(f_disp))) for _ in range(15)]
    plot_dispersion_fit("Demo Basin", f_disp, c_r0_obs, 0.04 * c_r0_obs, f_disp, c_best_dense, acc_curves, disp_fig)
    print(f"  Saved figures:\n    - {vs_fig}\n    - {disp_fig}")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
