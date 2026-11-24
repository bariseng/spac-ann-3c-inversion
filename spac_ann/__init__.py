"""Physics-guided 3-component SPAC/F-J, RayDec/DFA HVSR, PyTorch ADsurf determinant,
probabilistic deep-learning (MDN, JointNet, CNF), and Parallel-Tempering rj-MCMC
surface-wave inversion toolkit.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .adsurf_det import (
    invert_adsurf_joint,
    love_det_torch,
    newton_velocity_residual_torch,
    rayleigh_det_plucker_torch,
)
from .models_ann import (
    ConditionalNormalizingFlow3C,
    JointNetDispFormer,
    MDNInversionNet,
    synthesize_domain_aligned_hvsr,
)
from .physics import (
    F_GRID_DEFAULT,
    Z_GRID_DEFAULT,
    brocher_vp_rho,
    compute_profile_metrics,
    generate_synthetic_dataset,
    sample_basin_model,
    sh_transfer_f0,
)
from .raydec_dfa import compute_dfa_hvsr_full, compute_raydec_ellipticity
from .rjmcmc import gelman_rubin_rhat, run_parallel_tempering_rjmcmc
from .sac_io import SacHeader, geodesic_dist_m, read_sac, read_sac_header, slice_sac_window
from .spac_3c import (
    compute_3c_pair_coherence,
    compute_decoupled_3c_fj,
    estimate_sleeman_nsr,
    remove_near_dc_drift,
    rotate_pair_en_to_rt,
)

__all__ = [
    "__version__",
    "F_GRID_DEFAULT",
    "Z_GRID_DEFAULT",
    "SacHeader",
    "read_sac",
    "read_sac_header",
    "slice_sac_window",
    "geodesic_dist_m",
    "brocher_vp_rho",
    "sh_transfer_f0",
    "compute_profile_metrics",
    "sample_basin_model",
    "generate_synthetic_dataset",
    "remove_near_dc_drift",
    "estimate_sleeman_nsr",
    "rotate_pair_en_to_rt",
    "compute_3c_pair_coherence",
    "compute_decoupled_3c_fj",
    "compute_raydec_ellipticity",
    "compute_dfa_hvsr_full",
    "rayleigh_det_plucker_torch",
    "love_det_torch",
    "newton_velocity_residual_torch",
    "invert_adsurf_joint",
    "MDNInversionNet",
    "JointNetDispFormer",
    "ConditionalNormalizingFlow3C",
    "synthesize_domain_aligned_hvsr",
    "run_parallel_tempering_rjmcmc",
    "gelman_rubin_rhat",
]
