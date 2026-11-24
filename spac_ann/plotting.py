"""Publication-style Matplotlib plotting utilities for 3C SPAC / F-J / Vs(z) / dispersion fits."""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np
from scipy import special


def _format_log_ticks(ax: plt.Axes, axis: str, ticks: List[float]) -> None:
    loc = FixedLocator(ticks)
    labels = [f"{t:g}" for t in ticks]
    if axis == "x":
        ax.xaxis.set_major_locator(loc)
        ax.set_xticklabels(labels)
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(loc)
        ax.set_yticklabels(labels)
        ax.yaxis.set_minor_formatter(NullFormatter())


def plot_ring_spac_and_misfit(
    site_label: str,
    freqs: np.ndarray,
    rings: List[Dict],
    c_obs: np.ndarray,
    c_std: np.ndarray,
    valid: np.ndarray,
    max_ap: float,
    min_r: float,
    out_path: str,
) -> None:
    """Plot stacked ring-average SPAC coherency curves and 2D ring SPAC misfit map."""
    linestyles = ["-", "--", ":", "-.", (0, (6, 2, 1.5, 2)), (0, (8, 3))]
    c_map = np.geomspace(50.0, 2800.0, 180)
    r_rings = np.array([rg["r_m"] for rg in rings], dtype=np.float64)
    rho_rings = np.vstack([rg["rho_re"] for rg in rings])

    arg = (2.0 * np.pi * r_rings[:, None, None] * freqs[None, None, :]) / c_map[None, :, None]
    diff2 = (rho_rings[:, None, :] - special.j0(arg)) ** 2
    w_ring = np.where(arg <= 14.0, 1.0, 0.35)
    misfit = np.clip(
        np.sqrt(np.sum(w_ring * diff2, axis=0) / np.maximum(np.sum(w_ring, axis=0), 1e-6)) / 0.18,
        0.0,
        2.0,
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.5, 9.2), gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.24}
    )

    s1_cnt, s2_cnt = 0, 0
    for rg in rings:
        if rg.get("session", 1) == 1:
            ls, col = linestyles[s1_cnt % len(linestyles)], "#1f77b4"
            lbl = f"S1, r = {rg['r_m']:.0f} m"
            s1_cnt += 1
        else:
            ls, col = linestyles[s2_cnt % len(linestyles)], "#d96b27"
            lbl = f"S2, r = {rg['r_m']:.0f} m"
            s2_cnt += 1
        ax_top.plot(freqs, rg["rho_re"], linestyle=ls, color=col, lw=1.25, label=lbl)

    ax_top.axhline(0.0, color="#000000", lw=0.9)
    ax_top.set_xscale("log")
    ax_top.set_xlim(float(freqs.min()), float(freqs.max()))
    ax_top.set_ylim(-0.70, 1.10)
    _format_log_ticks(ax_top, "x", [0.25, 0.5, 1, 2, 5, 10])
    ax_top.grid(True, which="major", ls=":", color="#bfbfbf", lw=0.7)
    ax_top.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_top.set_ylabel("SPAC coefficient ρ", fontsize=11)
    ax_top.set_title(f"{site_label} — Ring Averages", fontsize=12, fontweight="bold")
    ax_top.legend(loc="upper right", fontsize=8, ncol=2 if len(rings) > 12 else 1, frameon=True, fancybox=False)

    pcm = ax_bot.pcolormesh(freqs, c_map, misfit, cmap="jet", vmin=0.0, vmax=2.0, shading="auto")
    ax_bot.set_xscale("log")
    ax_bot.set_yscale("log")
    ax_bot.set_xlim(float(freqs.min()), float(freqs.max()))
    ax_bot.set_ylim(50.0, 2800.0)
    _format_log_ticks(ax_bot, "x", [0.25, 0.5, 1, 2, 5, 10])
    _format_log_ticks(ax_bot, "y", [50, 100, 200, 500, 1000, 2000])
    ax_bot.plot(freqs, freqs * (2.0 * max_ap), "--", color="#ffffff", lw=1.4)
    ax_bot.plot(freqs, freqs * (2.0 * min_r), ":", color="#ffffff", lw=1.6)
    ax_bot.errorbar(
        freqs[valid], c_obs[valid], yerr=c_std[valid], fmt="o",
        color="#d62728", ecolor="#d62728", ms=4.3, capsize=2.5, elinewidth=1.1, zorder=5,
    )
    ax_bot.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_bot.set_ylabel("Phase velocity (m/s)", fontsize=11)
    ax_bot.set_title("Misfit of the rings (dashed: k_min, dotted: k_max)", fontsize=12, fontweight="bold")
    fig.colorbar(pcm, ax=ax_bot, pad=0.015, fraction=0.035)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def plot_vs_uncertainty(
    site_label: str,
    z_fine: np.ndarray,
    profs: np.ndarray,
    h_best: np.ndarray,
    vs_best: np.ndarray,
    d_max: float,
    vs30_best: float,
    vs100_best: float,
    out_path: str,
) -> None:
    """Plot classic 5-95% uncertainty Vs(z) profile, envelope, median, and best step model."""
    p5 = np.percentile(profs, 5, axis=0)
    p95 = np.percentile(profs, 95, axis=0)
    vmin = np.min(profs, axis=0)
    vmax = np.max(profs, axis=0)
    vmed = np.median(profs, axis=0)

    z_max = float(z_fine[-1])
    xs, ys = [], []
    z_cur = 0.0
    for k in range(len(vs_best)):
        z_next = min(z_cur + float(h_best[k]) if k < len(vs_best) - 1 else z_max, z_max)
        xs.extend([float(vs_best[k]), float(vs_best[k])])
        ys.extend([z_cur, z_next])
        z_cur = z_next
        if z_cur >= z_max - 1e-6:
            break

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.fill_betweenx(z_fine, p5, p95, color="#cfe2ff", label="5–95% acceptable models", zorder=1)
    ax.plot(vmin, z_fine, ":", color="#2c5aa0", lw=1.6, label="envelope (min–max)", zorder=2)
    ax.plot(vmax, z_fine, ":", color="#2c5aa0", lw=1.6, zorder=2)
    ax.plot(vmed, z_fine, "-", color="#0028cc", lw=2.2, label="median", zorder=3)
    ax.plot(xs, ys, "-", color="#e50000", lw=3.2, label="best model", zorder=4)
    ax.axhline(d_max, linestyle="--", color="#2b7bba", lw=1.8, label=f"Dmax = {d_max:.0f} m", zorder=5)
    ax.axvline(760.0, linestyle="--", color="#555555", lw=1.1, label="Vs = 760 m/s", zorder=2)

    x_max = float(max(1500.0, np.ceil(max(np.max(vmax), max(xs)) * 1.06 / 250.0) * 250.0))
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(z_max, 0.0)
    ax.grid(True, color="#d0d0d0", lw=0.7)
    ax.set_xlabel("Vs (m/s)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(f"{site_label} | Vs30 = {vs30_best:.0f} m/s, Vs100 = {vs100_best:.0f} m/s", fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True, fancybox=False, edgecolor="#000000")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def plot_dispersion_fit(
    site_label: str,
    f_obs: np.ndarray,
    c_obs: np.ndarray,
    c_std: np.ndarray,
    f_dense: np.ndarray,
    c_best_dense: np.ndarray,
    acceptable_curves: List[np.ndarray],
    out_path: str,
) -> None:
    """Plot log-log dispersion fit with acceptable gray curves, blue observations, and red best fit."""
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    for i, c_d in enumerate(acceptable_curves):
        ax.plot(
            f_dense, c_d, "-", color="#b8b8b8", lw=1.0, alpha=0.85,
            label=f"{len(acceptable_curves)} acceptable" if i == 0 else None, zorder=1,
        )
    ax.errorbar(
        f_obs, c_obs, yerr=c_std, fmt="o", color="#1e88e5", ecolor="#1e88e5",
        ms=6.8, capsize=4.5, elinewidth=1.4, label="Observed SPAC", zorder=3,
    )
    ax.plot(f_dense, c_best_dense, "-", color="#e50000", lw=3.2, label="best model", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", color="#d0d0d0", lw=0.7)
    ax.grid(True, which="minor", ls=":", color="#dcdcdc", lw=0.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Phase velocity (m/s)", fontsize=12)
    ax.set_title(f"{site_label} Dispersion: best (red), {len(acceptable_curves)} acceptable (grey)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True, fancybox=False, edgecolor="#000000")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
