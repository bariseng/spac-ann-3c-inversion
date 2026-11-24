"""Automated unit, analytical physics, differentiability, and bijection tests for ``spac_ann``.

All benchmarks are strictly self-contained and synthetic (no external data files required).
Run from repository root via::

    python tests/test_suite.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile

import numpy as np
import torch
from disba import PhaseDispersion
from scipy import signal, special

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spac_ann.adsurf_det import (
    invert_adsurf_joint,
    love_det_torch,
    newton_velocity_residual_torch,
    rayleigh_det_plucker_torch,
)
from spac_ann.models_ann import (
    ConditionalNormalizingFlow3C,
    JointNetDispFormer,
    MDNInversionNet,
    jointnet_loss,
    mdn_loss,
    synthesize_domain_aligned_hvsr,
)
from spac_ann.physics import (
    F_GRID_DEFAULT,
    Z_GRID_DEFAULT,
    brocher_vp_rho,
    compute_profile_metrics,
    sample_basin_model,
    sh_transfer_f0,
)
from spac_ann.raydec_dfa import compute_dfa_hvsr_full, compute_raydec_ellipticity
from spac_ann.rjmcmc import _rjmcmc_propose_move, gelman_rubin_rhat, run_parallel_tempering_rjmcmc
from spac_ann.sac_io import geodesic_dist_m, read_sac, slice_sac_window, to_local_xy
from spac_ann.spac_3c import (
    compute_3c_pair_coherence,
    compute_decoupled_3c_fj,
    estimate_array_nsr_and_correct,
    estimate_sleeman_nsr,
    konno_ohmachi_matrix,
    remove_near_dc_drift,
    rotate_pair_en_to_rt,
)


def _write_minimal_sac(path: str, data: np.ndarray, delta: float = 0.01, stla: float = 45.0, stlo: float = 10.0) -> None:
    """Create a valid little-endian binary SAC file for I/O roundtrip testing."""
    fhdr = np.full(70, -12345.0, dtype="<f4")
    ihdr = np.full(40, -12345, dtype="<i4")
    fhdr[0] = float(delta)
    fhdr[5] = 0.0
    fhdr[31] = float(stla)
    fhdr[32] = float(stlo)
    fhdr[33] = 0.0
    ihdr[0:6] = [2026, 1, 0, 0, 0, 0]
    ihdr[6] = 6  # header version
    ihdr[9] = len(data)
    chdr = b"-12345  " * 24
    with open(path, "wb") as f:
        f.write(fhdr.tobytes())
        f.write(ihdr.tobytes())
        f.write(chdr[:192])
        f.write(np.asarray(data, dtype="<f4").tobytes())


def main() -> None:
    passed = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal passed
        if not cond:
            raise AssertionError(f"FAILED: {msg}")
        passed += 1
        print(f"  [PASS] {msg}")

    print("=== 1. SAC I/O & Geodesic Utilities ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        sac_path = os.path.join(tmpdir, "synthetic_test.sac")
        x_raw = np.sin(np.linspace(0, 12.0 * np.pi, 1000)).astype(np.float32)
        _write_minimal_sac(sac_path, x_raw, delta=0.01, stla=45.0, stlo=10.0)
        hdr, tr = read_sac(sac_path)
        check(hdr.npts == 1000 and abs(hdr.fs - 100.0) < 1e-6, "SAC binary header parsed accurately")
        check(np.max(np.abs(tr - x_raw)) < 1e-6, "SAC trace float32 payload roundtrips identically")
        sl = slice_sac_window(hdr, tr, hdr.start_utc, hdr.start_utc + dt.timedelta(seconds=5.0))
        check(len(sl) == 500, "SAC time window slicing returns exact sample count")

    d_geo = geodesic_dist_m(45.0, 10.0, 45.0, 10.001)
    x_loc, y_loc = to_local_xy(np.array([45.0, 45.0]), np.array([10.0, 10.001]), 45.0, 10.0)
    check(75.0 <= d_geo <= 82.0 and abs(float(x_loc[1]) - d_geo) < 1.0 and abs(float(y_loc[1])) < 1.0,
          f"Geodesic & auto-UTM projection agree ({d_geo:.2f} m vs {x_loc[1]:.2f} m)")

    print("\n=== 2. 1D Elastodynamics & Basin Physics ===")
    h_1l = np.array([40.0, 0.0])
    vs_1l = np.array([240.0, 1600.0])
    rho_1l = np.array([1.85, 2.50])
    f0_1l, a0_1l = sh_transfer_f0(h_1l, vs_1l, rho_1l, np.geomspace(0.2, 10.0, 400))
    f0_ana = 240.0 / (4.0 * 40.0)  # 1.50 Hz
    check(abs(f0_1l - f0_ana) < 0.05 and a0_1l > 5.0,
          f"1D SH transfer resonance matches quarter-wavelength f0={f0_1l:.3f} Hz (exact={f0_ana:.3f})")

    h_s, vp_s, vs_s, rho_s = sample_basin_model(np.random.default_rng(101))
    check(5 <= len(h_s) <= 9 and len(h_s) == len(vs_s) and np.all(vp_s > vs_s * 1.49),
          f"Random basin model sampler yields {len(h_s)} layers with Vp > 1.49*Vs")
    met = compute_profile_metrics(h_s, vs_s, rho_s)
    check(np.isfinite(met["vs30"]) and met["vs30"] > 80.0 and len(met["vs_grid"]) == len(Z_GRID_DEFAULT),
          f"Profile engineering metrics valid (Vs30={met['vs30']:.1f} m/s)")

    print("\n=== 3. 3C SPAC/F-J Decoupling, Drift & NSR ===")
    r_syn = np.linspace(15.0, 500.0, 60)
    f_syn = 1.2
    cR_true, cL_true, aL_true = 400.0, 260.0, 0.60
    kr = 2.0 * np.pi * f_syn * r_syn / cR_true
    kl = 2.0 * np.pi * f_syn * r_syn / cL_true
    rho_tt_syn = aL_true * (special.j0(kl) - special.jv(2, kl)) + (1.0 - aL_true) * (special.j0(kr) + special.jv(2, kr))
    rho_rr_syn = aL_true * (special.j0(kl) + special.jv(2, kl)) + (1.0 - aL_true) * (special.j0(kr) - special.jv(2, kr))
    c_grid_3c = np.linspace(120.0, 800.0, 137)
    fj = compute_decoupled_3c_fj(
        np.array([f_syn, f_syn * 1.1]),
        c_grid_3c,
        r_syn,
        np.vstack([special.j0(kr), special.j0(kr)]).T,
        np.vstack([rho_rr_syn, rho_rr_syn]).T,
        np.vstack([rho_tt_syn, rho_tt_syn]).T,
    )
    cL_pk = float(c_grid_3c[int(np.argmax(fj["fj_love_norm"][0]))])
    cR_pk = float(c_grid_3c[int(np.argmax(fj["fj_rr_norm"][0]))])
    check(abs(cL_pk - cL_true) <= 15.0 and abs(cR_pk - cR_true) <= 15.0,
          f"3C decoupled F-J isolates Love cL={cL_pk:.0f} (true={cL_true:.0f}) & Rayleigh cR={cR_pk:.0f} (true={cR_true:.0f})")

    # Isotropic Z-R / R-Z antisymmetric coherency must be purely real
    rng_zr = np.random.default_rng(42)
    fs_zr, n_zr = 50.0, 4096 * 8
    freqs_zr = np.fft.rfftfreq(n_zr, 1.0 / fs_zr)
    sos_zr = signal.butter(4, [1.2, 1.8], btype="bandpass", fs=fs_zr, output="sos")
    e1 = np.zeros(n_zr); n1 = np.zeros(n_zr); z1 = np.zeros(n_zr)
    e2 = np.zeros(n_zr); n2 = np.zeros(n_zr); z2 = np.zeros(n_zr)
    for az in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
        wf = signal.sosfiltfilt(sos_zr, rng_zr.normal(0.0, 1.0, n_zr))
        dly = 80.0 * np.cos(az) / 300.0
        W = np.fft.rfft(wf)
        wj = np.fft.irfft(W * np.exp(-1j * 2.0 * np.pi * freqs_zr * dly), n=n_zr)
        ri_p = np.fft.irfft(1j * 1.8 * W, n=n_zr)
        rj_p = np.fft.irfft(1j * 1.8 * W * np.exp(-1j * 2.0 * np.pi * freqs_zr * dly), n=n_zr)
        z1 += wf; z2 += wj
        n1 += ri_p * np.cos(az); e1 += ri_p * np.sin(az)
        n2 += rj_p * np.cos(az); e2 += rj_p * np.sin(az)
    ko_mat = konno_ohmachi_matrix(freqs_zr, np.array([1.5]), b=40.0)
    coh_3c = compute_3c_pair_coherence(
        np.fft.rfft(e1)[None, :], np.fft.rfft(n1)[None, :], np.fft.rfft(z1)[None, :],
        np.fft.rfft(e2)[None, :], np.fft.rfft(n2)[None, :], np.fft.rfft(z2)[None, :],
        az_rad=0.0, ko_mat=ko_mat,
    )
    check(float(coh_3c["rho_zr_anti"][0]) > 0.50,
          f"3C Z-R/R-Z real anti-coherence signal non-zero ({coh_3c['rho_zr_anti'][0]:+.3f})")

    # Thermal/near-DC drift suppression
    fs_d = 100.0
    t_s = np.arange(0, 400.0, 1.0 / fs_d)
    sig_1hz = np.sin(2.0 * np.pi * 1.0 * t_s)
    cln, diag = remove_near_dc_drift(sig_1hz + 10.0 * np.sin(2.0 * np.pi * 0.05 * t_s), fs_d, cutoff_hz=0.15)
    amp_ratio = float(np.std(cln[1500:-1500]) / np.std(sig_1hz[1500:-1500]))
    check(diag["drift_suppression_db"] > 25.0 and abs(amp_ratio - 1.0) < 0.03,
          f"Near-DC drift suppressed {diag['drift_suppression_db']:.1f} dB, 1.0 Hz ratio={amp_ratio:.4f}")

    # NSR white-noise edge case & Sleeman 3-station check
    rng = np.random.default_rng(7)
    nsr_wn = estimate_array_nsr_and_correct(
        r_arr=np.linspace(20.0, 300.0, 21),
        rho_zz_re=rng.normal(0.0, 0.03, size=(21, 64)),
        rho_zz_im=rng.normal(0.0, 0.03, size=(21, 64)),
        rho_zz_se=np.full((21, 64), 0.05),
        c_base=np.full(64, 400.0),
    )
    check(bool(np.all(nsr_wn["noise_dominated"])) and np.max(np.abs(nsr_wn["rho_zz_corr"])) < 0.25,
          "Incoherent white noise flagged noise_dominated=True and not amplified")

    ko_s = konno_ohmachi_matrix(np.fft.rfftfreq(2048, 0.01), F_GRID_DEFAULT, b=40.0)
    s_comm = np.fft.rfft(rng.normal(0.0, 1.0, (24, 2048)), axis=1)

    def _build_csd(sigmas):
        traces = [s_comm + np.fft.rfft(rng.normal(0.0, sg, (24, 2048)), axis=1) for sg in sigmas]
        csd = np.zeros((3, 3, len(F_GRID_DEFAULT)), dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                csd[i, j] = ko_s @ np.mean(traces[i] * np.conj(traces[j]), axis=0)
        return csd

    nsr_clean_sl = estimate_sleeman_nsr(_build_csd([0.1, 0.1, 0.1]))
    nsr_noisy_sl = estimate_sleeman_nsr(_build_csd([0.6, 0.6, 0.6]))
    check(float(np.nanmedian(nsr_noisy_sl)) > float(np.nanmedian(nsr_clean_sl)) * 5.0,
          f"Sleeman 3-station self-noise detects noisy triplet ({np.nanmedian(nsr_noisy_sl):.3f} vs {np.nanmedian(nsr_clean_sl):.3f})")

    print("\n=== 4. Hobiger RayDec & Elastic DFA Analytical Tests ===")
    fs_rd, dur_rd = 50.0, 3600.0
    n_rd = int(dur_rd * fs_rd)
    t_rd = np.arange(n_rd) / fs_rd

    def _make_transient_mix(e_true: float, l_amp: float, seed_v: int):
        rr = np.random.default_rng(seed_v)
        zs = rr.normal(0.0, 0.05, n_rd)
        ns = rr.normal(0.0, 0.05, n_rd)
        es = rr.normal(0.0, 0.05, n_rd)
        for _ in range(550):
            t0 = rr.uniform(20.0, dur_rd - 20.0)
            fc = rr.uniform(0.5, 3.0)
            st = rr.uniform(2.0, 5.0) / fc
            az = rr.uniform(0.0, 2.0 * np.pi)
            amp = rr.uniform(0.5, 2.0)
            i0, i1 = max(0, int((t0 - 4.0 * st) * fs_rd)), min(n_rd, int((t0 + 4.0 * st) * fs_rd))
            tt = t_rd[i0:i1] - t0
            env = np.exp(-0.5 * (tt / st) ** 2)
            ph = 2.0 * np.pi * fc * tt + rr.uniform(0.0, 2.0 * np.pi)
            zs[i0:i1] += amp * env * np.cos(ph)
            rp = -e_true * amp * env * np.sin(ph)
            ns[i0:i1] += rp * np.cos(az)
            es[i0:i1] += rp * np.sin(az)
        for _ in range(550):
            t0 = rr.uniform(20.0, dur_rd - 20.0)
            fc = rr.uniform(0.5, 3.0)
            st = rr.uniform(2.0, 5.0) / fc
            az = rr.uniform(0.0, 2.0 * np.pi)
            amp = l_amp * rr.uniform(0.5, 2.0)
            i0, i1 = max(0, int((t0 - 4.0 * st) * fs_rd)), min(n_rd, int((t0 + 4.0 * st) * fs_rd))
            tt = t_rd[i0:i1] - t0
            env = np.exp(-0.5 * (tt / st) ** 2)
            ph = 2.0 * np.pi * fc * tt + rr.uniform(0.0, 2.0 * np.pi)
            lp = amp * env * np.cos(ph)
            ns[i0:i1] += lp * (-np.sin(az))
            es[i0:i1] += lp * np.cos(az)
        return es, ns, zs

    e_m1, n_m1, z_m1 = _make_transient_mix(2.0, 2.0, 123)
    rd_m1 = compute_raydec_ellipticity(e_m1, n_m1, z_m1, fs=fs_rd, freqs=np.array([1.0]), chunk_sec=600.0)
    check(1.80 <= float(rd_m1["ellip_raydec"][0]) <= 2.30,
          f"Hobiger RayDec rejects equal Love wave: E(1Hz)={rd_m1['ellip_raydec'][0]:.3f} (true=2.00)")

    _, hv_hs_tot, _, _ = compute_dfa_hvsr_full(
        np.array([0.0]), np.array([1000.0 * np.sqrt(3.0)]), np.array([1000.0]), np.array([2.2]),
        np.array([0.5, 1.0, 2.0, 5.0]), damping=0.012,
    )
    check(bool(np.all(np.abs(hv_hs_tot - 1.344) < 0.04)),
          f"Elastic DFA Poisson half-space HV_tot={hv_hs_tot[0]:.3f} (Sanchez-Sesma 1.344)")

    f_b = np.array([0.02, 0.25, 0.50, 0.625, 0.75, 1.5, 5.0])
    hv_b, _, _, _ = compute_dfa_hvsr_full(
        np.array([100.0, 0.0]), np.array([1550.0, 3000.0]), np.array([250.0, 1500.0]), np.array([1.85, 2.45]),
        f_b, damping=0.010,
    )
    hv_b_ref, _, _, _ = compute_dfa_hvsr_full(
        np.array([100.0, 0.0]), np.array([1550.0, 3000.0]), np.array([250.0, 1500.0]), np.array([1.85, 2.45]),
        f_b, damping=0.010, n_k_core=2400, n_k_tail=160,
    )
    dfa_quad_err = float(np.max(np.abs(hv_b - hv_b_ref) / hv_b_ref) * 100.0)
    check(hv_b[0] < 1.08 and abs(f_b[int(np.argmax(hv_b))] - 0.625) < 1e-6 and dfa_quad_err < 1.5,
          f"Elastic DFA 1-layer peaks at f0=0.625 Hz & quadrature converges within {dfa_quad_err:.2f}%")

    print("\n=== 5. PyTorch ADsurf Secular Determinants & Joint Inversion ===")
    h_t = torch.tensor([15.0, 35.0, 80.0, 120.0, 0.0], dtype=torch.float64)
    vs_t = torch.tensor([180.0, 310.0, 480.0, 390.0, 1300.0], dtype=torch.float64, requires_grad=True)
    vp_t = torch.tensor([1550.0, 1650.0, 1850.0, 1750.0, 2800.0], dtype=torch.float64)
    rho_t = torch.tensor([1.80, 1.90, 2.05, 2.00, 2.35], dtype=torch.float64)
    vm = np.column_stack([h_t.numpy() / 1000.0, vp_t.numpy() / 1000.0, vs_t.detach().numpy() / 1000.0, rho_t.numpy()])
    pd_c = PhaseDispersion(*vm.T)
    t_chk = (1.0 / np.array([0.6, 1.2, 2.5, 5.0]))[::-1].copy()
    for wave in ("rayleigh", "love"):
        for mode in (0, 1):
            r = pd_c(t_chk, mode=mode, wave=wave)
            fm = torch.tensor(1.0 / r.period, dtype=torch.float64)
            cm = torch.tensor(r.velocity * 1000.0, dtype=torch.float64)
            if wave == "rayleigh":
                d0 = torch.abs(rayleigh_det_plucker_torch(fm, cm, h_t, vp_t, vs_t, rho_t))
                res_off = newton_velocity_residual_torch(rayleigh_det_plucker_torch, fm, cm * 1.03, h_t, vp_t, vs_t, rho_t)
            else:
                d0 = torch.abs(love_det_torch(fm, cm, h_t, vs_t, rho_t))
                res_off = newton_velocity_residual_torch(love_det_torch, fm, cm * 1.03, h_t, vs_t, rho_t)
            loss = (res_off**2).mean()
            loss.backward()
            gn = float(vs_t.grad.norm().item())
            vs_t.grad.zero_()
            check(float(d0.max().item()) < 1e-4 and np.isfinite(gn) and gn > 1e-8,
                  f"PyTorch {wave} mode={mode}: |D(c_exact)|={d0.max().item():.2e}, |dL/dVs|={gn:.2e}")

    f_inv = np.geomspace(0.4, 6.0, 24)
    t_inv_asc = (1.0 / f_inv)[::-1].copy()
    vp_b_syn, rho_b_syn = brocher_vp_rho(h_t.numpy(), vs_t.detach().numpy(), z_wt=6.0)
    pd_syn = PhaseDispersion(*np.column_stack([
        h_t.numpy() / 1000.0, vp_b_syn / 1000.0, vs_t.detach().numpy() / 1000.0, rho_b_syn
    ]).T)
    c_r0_syn = (pd_syn(t_inv_asc, mode=0, wave="rayleigh").velocity * 1000.0)[::-1].copy()
    c_l0_syn = (pd_syn(t_inv_asc, mode=0, wave="love").velocity * 1000.0)[::-1].copy()
    f0_syn, _ = sh_transfer_f0(h_t.numpy(), vs_t.detach().numpy(), rho_b_syn, np.geomspace(0.2, 12.0, 160))
    inv_out = invert_adsurf_joint(
        h_init_base=np.array([18.0, 40.0, 75.0, 110.0, 0.0]),
        vs_init_base=np.array([210.0, 340.0, 520.0, 430.0, 1180.0]),
        f_disp=f_inv,
        c_r0_obs=c_r0_syn,
        c_l0_obs=c_l0_syn,
        f0_obs=f0_syn,
        hvsr_reliable=True,
        d_max_m=320.0,
        n_starts=2,
        n_steps_stage1=70,
        run_stage2=True,
    )
    vm_inv = np.column_stack([
        inv_out["h_opt"] / 1000.0,
        brocher_vp_rho(inv_out["h_opt"], inv_out["vs_opt"])[0] / 1000.0,
        inv_out["vs_opt"] / 1000.0,
        brocher_vp_rho(inv_out["h_opt"], inv_out["vs_opt"])[1],
    ])
    c_r0_rec = (PhaseDispersion(*vm_inv.T)(t_inv_asc, mode=0, wave="rayleigh").velocity * 1000.0)[::-1]
    r0_rec_err = float(np.mean(np.abs(c_r0_rec - c_r0_syn) / c_r0_syn) * 100.0)
    check(r0_rec_err < 3.5, f"ADsurf Stage-1+2 recovers R0 dispersion within {r0_rec_err:.2f}% (<3.5%)")

    print("\n=== 6. ANN, JointNet & 3C Normalizing Flow Bijection ===")
    torch.manual_seed(0)
    mdn = MDNInversionNet()
    jnet = JointNetDispFormer()
    d4, h3, s6 = torch.randn(4, 4, 64), torch.randn(4, 3, 64), torch.randn(4, 6)
    tgt = torch.randn(4, 53)
    tgt[:, 48:] = torch.clamp(tgt[:, 48:], 0.0, 1.0)
    log_pi, mu, sigma, cls_m = mdn(d4, h3, s6)
    lm = mdn_loss(log_pi, mu, sigma, cls_m, tgt)
    vp_j, zp_j, cls_j = jnet(d4, h3, s6)
    lj = jointnet_loss(vp_j, zp_j, cls_j, tgt)
    hv_dom, f0_dom, a0_dom, hvv_dom = synthesize_domain_aligned_hvsr(
        F_GRID_DEFAULT,
        np.array([1.2, 2.0]),
        np.array([4.5, 6.0]),
        np.full((2, len(F_GRID_DEFAULT)), 1.8),
        np.random.default_rng(1),
    )
    check(torch.isfinite(lm) and torch.isfinite(lj) and hv_dom.shape == (2, 3, len(F_GRID_DEFAULT)) and len(f0_dom) == 2 and len(a0_dom) == 2 and len(hvv_dom) == 2,
          "MDN & JointNet forward/loss + domain-aligned HVSR shapes verified")

    cnf = ConditionalNormalizingFlow3C()
    cnf.eval()
    with torch.no_grad():
        y_in = torch.randn(6, 51)
        cond = cnf.encode(torch.randn(6, 7, 64), torch.randn(6, 4, 64), torch.randn(6, 8))
        z_fwd = y_in.clone()
        for fl in cnf.flows:
            z_fwd, _ = fl(z_fwd, cond)
        y_rec = z_fwd.clone()
        for fl in reversed(cnf.flows):
            y_rec = fl.inverse(y_rec, cond)
        bij_err = float(torch.max(torch.abs(y_rec - y_in)).item())
        lp_cnf, _ = cnf.log_prob(y_in, torch.randn(6, 7, 64), torch.randn(6, 4, 64), torch.randn(6, 8))
    check(bij_err < 1e-4 and bool(torch.all(torch.isfinite(lp_cnf))),
          f"3C CNF exact bijection max_inv_err={bij_err:.2e} (<1e-4)")

    print("\n=== 7. Trans-Dimensional rj-MCMC Detailed Balance & PT Run ===")
    rng_db = np.random.default_rng(77)
    z_db = np.array([20.0, 60.0, 150.0, 320.0])
    vs_db = np.array([200.0, 350.0, 500.0, 800.0, 1200.0])
    k_db = []
    for it_db in range(70000):
        pz, pvs, lqr = _rjmcmc_propose_move(z_db, vs_db, rng_db, k_min=3, k_max=10)
        if pz is not None and pvs is not None:
            if lqr >= 0.0 or np.log(rng_db.random()) < lqr:
                z_db, vs_db = pz, pvs
        if it_db >= 10000:
            k_db.append(len(vs_db))
    kv_db, kc_db = np.unique(k_db, return_counts=True)
    pr_db = dict(zip(kv_db.tolist(), (kc_db / len(k_db)).tolist()))
    ratio_10_9 = pr_db.get(10, 0.0) / max(pr_db.get(9, 1e-9), 1e-9)
    check(0.55 <= ratio_10_9 <= 1.15,
          f"rj-MCMC detailed balance prior-only test: P(k=10)/P(k=9)={ratio_10_9:.3f}")

    rj_mini = run_parallel_tempering_rjmcmc(
        h_init=inv_out["h_opt"],
        vs_init=inv_out["vs_opt"],
        f_r0=f_inv,
        c_r0=c_r0_syn,
        s_r0=0.05 * c_r0_syn,
        f_l0=f_inv,
        c_l0=c_l0_syn,
        s_l0=0.07 * c_l0_syn,
        f0_obs=0.85,
        hvsr_reliable=True,
        d_max_m=320.0,
        log_vs_deep_prior=np.log(np.full(len(Z_GRID_DEFAULT), 1100.0)),
        n_chains=2,
        n_iter=300,
        burn_in=100,
        thin=2,
        seed=99,
    )
    rhat_conv = gelman_rubin_rhat([np.ones(50) + np.random.default_rng(i).normal(0, 0.01, 50) for i in range(4)])
    check(rj_mini["n_samples"] > 50 and np.isfinite(rj_mini["rhat_z760"]) and rhat_conv < 1.1,
          f"Parallel-Tempering rj-MCMC sampled {rj_mini['n_samples']} posteriors, Rhat_conv={rhat_conv:.3f}")

    print(f"\nALL CHECKS PASSED ({passed}/{passed}).")


if __name__ == "__main__":
    main()
