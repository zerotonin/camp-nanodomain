#!/usr/bin/env python3
"""
cAMP Nanodomain Model — Parameter Optimisation
================================================

Fits the 1D reaction-diffusion model of cAMP compartmentalisation in
Drosophila Kenyon cell gamma lobes to experimental ΔF/F imaging data.

Both genotypes (dnc-wt and dnc-KD) are fit SIMULTANEOUSLY with shared
biophysical parameters; only PDE activity differs between genotypes.

Usage:
    python optimise_model.py                    # full optimisation
    python optimise_model.py --quick            # quick test (fewer iterations)
    python optimise_model.py --plot-only best_params.json   # plot existing fit

Requirements:
    numpy, scipy, pandas, matplotlib
    (all standard scientific Python)

Author: [Your Lab]
Date:   2026
"""

import numpy as np
import json
import argparse
import os
import sys
import time as time_module
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional

import pandas as pd
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
from scipy.optimize import differential_evolution, minimize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# 1. MODEL DEFINITION (self-contained)
# =============================================================================

# --- Geometry ---

BOUTON = 0
AXON = 1


@dataclass
class Geometry:
    """1D spatial grid for the KC axon."""
    n_compartments: int = 5
    bouton_diameter: float = 1.5       # μm
    axon_diameter: float = 0.3         # μm
    inter_bouton_distance: float = 5.0 # μm
    axon_bins_per_segment: int = 20

    # Built by build()
    x: np.ndarray = field(default=None, repr=False)
    dx: np.ndarray = field(default=None, repr=False)
    A: np.ndarray = field(default=None, repr=False)
    element_type: np.ndarray = field(default=None, repr=False)
    compartment_id: np.ndarray = field(default=None, repr=False)
    bouton_indices: list = field(default=None, repr=False)
    n_total: int = 0

    def build(self):
        """Construct the spatial grid."""
        N = self.n_compartments
        n_ax = self.axon_bins_per_segment
        d_b = self.bouton_diameter
        d_a = self.axon_diameter
        A_b = np.pi * (d_b / 2) ** 2
        A_a = np.pi * (d_a / 2) ** 2
        axon_len = self.inter_bouton_distance - d_b
        ax_dx = axon_len / n_ax

        positions, dx_arr, areas, types, comp_ids = [], [], [], [], []
        bouton_idx = []
        cx = 0.0
        idx = 0

        for i in range(N):
            # Bouton
            bouton_idx.append(idx)
            positions.append(cx + d_b / 2)
            dx_arr.append(d_b)
            areas.append(A_b)
            types.append(BOUTON)
            comp_ids.append(i)
            idx += 1
            cx += d_b

            # Axon segment
            if i < N - 1:
                for j in range(n_ax):
                    positions.append(cx + (j + 0.5) * ax_dx)
                    dx_arr.append(ax_dx)
                    areas.append(A_a)
                    types.append(AXON)
                    comp_ids.append(-1)
                    idx += 1
                cx += axon_len

        self.x = np.array(positions)
        self.dx = np.array(dx_arr)
        self.A = np.array(areas)
        self.element_type = np.array(types, dtype=int)
        self.compartment_id = np.array(comp_ids, dtype=int)
        self.bouton_indices = bouton_idx
        self.n_total = idx
        return self


# --- Parameters ---

@dataclass
class FittableParams:
    """Parameters that can be optimised.

    All biophysical parameters shared between WT and dunce;
    only PDE activity (k_cat, PDE_conc) differs by genotype.
    """
    # Diffusion & buffering
    D_free: float = 130.0        # μm²/s  (free cAMP diffusion)
    B_total: float = 20.0        # μM     (buffer capacity)
    K_D_buffer: float = 2.0      # μM     (buffer Kd)
    k_on_buffer: float = 10.0    # μM⁻¹s⁻¹

    # Adenylyl cyclase
    V_basal: float = 0.5         # μM/s   (basal production)
    V_max_AC: float = 10.0       # μM/s   (Ca²⁺-stimulated max)
    K_Ca: float = 0.5            # μM     (Ca²⁺ half-activation)
    n_Hill_AC: float = 2.0       # Hill coefficient

    # PDE (dunce)
    k_cat: float = 5.0           # s⁻¹    (catalytic rate)
    K_m: float = 2.4             # μM     (Michaelis constant)
    PDE_conc: float = 1.0        # μM     (PDE concentration in boutons)

    # Calcium
    Ca_rest: float = 0.05        # μM
    Ca_amplitude: float = 2.0    # μM     (peak Ca²⁺ per shock)
    tau_Ca: float = 1.0          # s      (Ca²⁺ clearance)
    Ca_pulse_dur: float = 0.5    # s      (Ca²⁺ influx duration)

    # Observation model: dF/F = alpha * c_free + dff_offset
    alpha: float = 0.05          # dF/F per μM cAMP
    dff_offset: float = 0.0      # baseline dF/F offset

    @property
    def k_off_buffer(self):
        return self.k_on_buffer * self.K_D_buffer

    @property
    def V_max_PDE(self):
        return self.PDE_conc * self.k_cat


# --- Stimulation ---

@dataclass
class StimProtocol:
    """Training protocol — times of odor-shock pairings.

    Each pairing: odor for `odor_dur` seconds, shock during last `shock_dur`.
    Ca²⁺ influx in aversive compartments (γ1, γ2) during shock.
    AC enhancement in ALL boutons during odor (KC is activated by odor).
    """
    # Pairing onset times (s) — relative to t=0
    pairing_onsets: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 30.0, 60.0, 90.0, 120.0])
    )
    odor_dur: float = 5.0         # s
    shock_onset_in_trial: float = 4.0  # s after odor onset
    shock_dur: float = 1.0        # s
    aversive_comps: list = field(default_factory=lambda: [0, 1])  # γ1, γ2


# --- Core dynamics ---

def build_coupling(geom: Geometry, D: float):
    """Compute diffusive coupling coefficients between grid bins."""
    n = geom.n_total
    alpha_p = np.zeros(n)
    alpha_m = np.zeros(n)
    for i in range(n - 1):
        A_int = 2.0 * geom.A[i] * geom.A[i+1] / (geom.A[i] + geom.A[i+1])
        dist = 0.5 * (geom.dx[i] + geom.dx[i+1])
        flux = D * A_int / dist
        V_i = geom.A[i] * geom.dx[i]
        V_ip1 = geom.A[i+1] * geom.dx[i+1]
        alpha_p[i] = flux / V_i
        alpha_m[i+1] = flux / V_ip1
    return alpha_p, alpha_m


def make_rhs(geom: Geometry, params: FittableParams, stim: StimProtocol,
             is_dunce: bool):
    """Build the ODE right-hand-side function.

    State vector: [c_free (N), c_bound (N), Ca²⁺ (N)]
    """
    N = geom.n_total
    alpha_p, alpha_m = build_coupling(geom, params.D_free)
    is_bouton = (geom.element_type == BOUTON)

    # PDE activity: only in boutons, zero in axon; zero everywhere for dunce
    V_max_pde = np.zeros(N)
    if not is_dunce:
        V_max_pde[is_bouton] = params.V_max_PDE

    # Precompute enzyme params
    K_m = params.K_m
    K_Ca = params.K_Ca
    n_H = params.n_Hill_AC
    V_bas = params.V_basal
    V_ac = params.V_max_AC
    k_on = params.k_on_buffer
    k_off = params.k_off_buffer
    B_tot = params.B_total
    Ca_rest = params.Ca_rest
    tau_Ca = params.tau_Ca
    Ca_amp = params.Ca_amplitude
    Ca_dur = params.Ca_pulse_dur

    # Precompute which boutons are aversive targets
    aversive_mask = np.zeros(N, dtype=bool)
    for idx in geom.bouton_indices:
        c = geom.compartment_id[idx]
        if c in stim.aversive_comps:
            aversive_mask[idx] = True

    # Pairing timing arrays
    pairing_onsets = stim.pairing_onsets
    odor_dur = stim.odor_dur
    shock_offset = stim.shock_onset_in_trial
    shock_dur = stim.shock_dur

    def rhs(t, y):
        c = y[:N]
        cb = y[N:2*N]
        ca = y[2*N:3*N]

        dc = np.zeros(N)
        dcb = np.zeros(N)
        dca = np.zeros(N)

        # 1. Diffusion
        dc[:-1] += alpha_p[:-1] * (c[1:] - c[:-1])
        dc[1:]  += alpha_m[1:]  * (c[:-1] - c[1:])

        c_pos = np.maximum(c, 0.0)

        # 2. AC production (boutons only, during odor)
        odor_on = False
        for onset in pairing_onsets:
            if onset <= t < onset + odor_dur:
                odor_on = True
                break

        for bi in geom.bouton_indices:
            # Basal
            dc[bi] += V_bas
            # Ca²⁺-stimulated (during odor)
            if odor_on:
                f_ca = ca[bi]**n_H / (K_Ca**n_H + ca[bi]**n_H + 1e-30)
                dc[bi] += V_ac * f_ca

        # 3. PDE degradation
        dc -= V_max_pde * c_pos / (K_m + c_pos + 1e-30)

        # 4. Buffering
        B_free = np.maximum(B_tot - cb, 0.0)
        j_bind = k_on * c_pos * B_free
        j_unbind = k_off * np.maximum(cb, 0.0)
        dc  += -j_bind + j_unbind
        dcb +=  j_bind - j_unbind

        # 5. Calcium
        dca -= (ca - Ca_rest) / tau_Ca

        # Ca²⁺ influx during shock in aversive boutons
        for onset in pairing_onsets:
            shock_start = onset + shock_offset
            if shock_start <= t < shock_start + shock_dur:
                dca[aversive_mask] += Ca_amp / Ca_dur
                break

        return np.concatenate([dc, dcb, dca])

    return rhs


def run_model(params: FittableParams, stim: StimProtocol,
              is_dunce: bool = False,
              geom: Optional[Geometry] = None,
              t_eval: Optional[np.ndarray] = None,
              t_end: float = 180.0,
              ) -> Tuple[np.ndarray, np.ndarray, Geometry]:
    """Run the model and return (time, bouton_c_free[n_t, 5], geom).

    Returns the free cAMP concentration in each of the 5 boutons.
    """
    if geom is None:
        geom = Geometry().build()

    N = geom.n_total
    rhs = make_rhs(geom, params, stim, is_dunce)

    y0 = np.zeros(3 * N)
    y0[2*N:3*N] = params.Ca_rest  # Ca²⁺ at rest

    if t_eval is None:
        t_eval = np.arange(0.0, t_end, 0.25)

    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), y0,
        method='BDF', t_eval=t_eval,
        max_step=0.1, rtol=1e-6, atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Extract bouton concentrations
    c_free = sol.y[:N, :].T  # [n_time, N_grid]
    bouton_c = np.column_stack([
        c_free[:, geom.bouton_indices[k]] for k in range(5)
    ])

    return sol.t, bouton_c, geom


def model_to_dff(bouton_c_free: np.ndarray, params: FittableParams) -> np.ndarray:
    """Convert model cAMP (μM) to predicted ΔF/F."""
    return params.alpha * bouton_c_free + params.dff_offset


# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(csv_path: str) -> Dict:
    """Load and preprocess the experimental data.

    Returns a dict with keys like ('dnc-wt', 'g2') → (time, mean_dff, sem_dff).
    Time is shifted so that the first pairing onset ≈ 0.
    Only uses Bouton structure data.
    """
    df = pd.read_csv(csv_path)
    bouton = df[df['structure'] == 'Bouton'].copy()

    # Per-fly average first (avoids double-counting multiple ROIs per fly)
    fly_avg = bouton.groupby(
        ['genotype', 'compartment', 'fly_id', 'time_s']
    )['dff'].mean().reset_index()

    # Across-fly mean and SEM
    summary = fly_avg.groupby(
        ['genotype', 'compartment', 'time_s']
    )['dff'].agg(['mean', 'sem', 'count']).reset_index()

    data = {}
    for geno in ['dnc-wt', 'dnc-KD']:
        for comp in ['g1', 'g2', 'g3', 'g4', 'g5']:
            sub = summary[
                (summary['genotype'] == geno) &
                (summary['compartment'] == comp)
            ].sort_values('time_s')

            if len(sub) == 0:
                continue

            t = sub['time_s'].values
            mean_dff = sub['mean'].values
            sem_dff = sub['sem'].values
            # Replace zero/NaN SEM with median SEM to avoid div-by-zero
            sem_dff = np.where(
                np.isfinite(sem_dff) & (sem_dff > 0),
                sem_dff,
                np.nanmedian(sem_dff[sem_dff > 0])
            )
            data[(geno, comp)] = (t, mean_dff, sem_dff)

    return data


def downsample_data(data: Dict, dt_target: float = 1.0) -> Dict:
    """Downsample data to reduce cost function evaluations.

    Averages ΔF/F in bins of width dt_target.
    """
    out = {}
    for key, (t, m, s) in data.items():
        t_bins = np.arange(t.min(), t.max(), dt_target)
        m_ds = np.array([m[(t >= tb) & (t < tb + dt_target)].mean()
                         for tb in t_bins])
        s_ds = np.array([s[(t >= tb) & (t < tb + dt_target)].mean()
                         for tb in t_bins])
        t_ds = t_bins + dt_target / 2
        mask = np.isfinite(m_ds)
        out[key] = (t_ds[mask], m_ds[mask], s_ds[mask])
    return out


# =============================================================================
# 3. COST FUNCTION
# =============================================================================

# Compartment index mapping
COMP_MAP = {'g1': 0, 'g2': 1, 'g3': 2, 'g4': 3, 'g5': 4}


def cost_function(x: np.ndarray, data: Dict, stim: StimProtocol,
                  geom: Geometry, param_names: List[str],
                  base_params: FittableParams,
                  fit_compartments: List[str] = None,
                  weights: Dict = None,
                  ) -> float:
    """Compute weighted sum-of-squares residual.

    Runs the model for BOTH genotypes with the same parameters
    (differing only in PDE activity) and compares to data.

    Args:
        x: Parameter vector (values for param_names)
        data: Experimental data dict
        stim: Stimulation protocol
        geom: Prebuilt geometry
        param_names: Which parameters x corresponds to
        base_params: Default values for unfitted parameters
        fit_compartments: Which compartments to include (default: all)
        weights: Per-compartment weights dict, e.g. {'g4': 2.0, 'g5': 2.0}
    """
    # Unpack parameters
    params = FittableParams(**{k: getattr(base_params, k)
                               for k in base_params.__dataclass_fields__})
    for name, val in zip(param_names, x):
        setattr(params, name, val)

    if fit_compartments is None:
        fit_compartments = ['g1', 'g2', 'g3', 'g4', 'g5']

    if weights is None:
        weights = {}

    # Common time grid (post-stimulus only: 0 to 180s)
    t_eval = np.arange(0.0, 180.0, 1.0)

    total_cost = 0.0
    n_points = 0

    for geno_label, is_dunce in [('dnc-wt', False), ('dnc-KD', True)]:
        try:
            t_model, bouton_c, _ = run_model(
                params, stim, is_dunce=is_dunce, geom=geom, t_eval=t_eval
            )
            pred_dff = model_to_dff(bouton_c, params)
        except Exception:
            return 1e12  # penalise failed integrations

        for comp in fit_compartments:
            key = (geno_label, comp)
            if key not in data:
                continue

            t_data, m_data, s_data = data[key]

            # Only use post-stimulus data (t >= 0)
            mask = t_data >= 0.0
            t_d = t_data[mask]
            m_d = m_data[mask]
            s_d = s_data[mask]

            # Interpolate model to data time points
            comp_idx = COMP_MAP[comp]
            pred_at_data = np.interp(t_d, t_model, pred_dff[:, comp_idx])

            # Weighted residual: (pred - obs)² / sem²
            w = weights.get(comp, 1.0)
            residuals = (pred_at_data - m_d) ** 2 / (s_d ** 2 + 1e-6)
            total_cost += w * np.sum(residuals)
            n_points += len(t_d)

    # Normalise by number of data points
    if n_points > 0:
        total_cost /= n_points

    return total_cost


# =============================================================================
# 4. OPTIMISATION CONFIGURATION
# =============================================================================

# --- Define which parameters to fit and their bounds ---

# This is the main configuration section. Adjust as needed.
# Parameters not listed here are held at their default values.

PARAM_CONFIG = {
    # name             : (lower_bound, upper_bound, default)
    'V_basal'          : (0.01,   5.0,     0.5),
    'V_max_AC'         : (1.0,   50.0,    10.0),
    'k_cat'            : (1.0,  200.0,     5.0),
    'PDE_conc'         : (0.1,   10.0,     1.0),
    'Ca_amplitude'     : (0.5,   20.0,     2.0),
    'tau_Ca'           : (0.2,    5.0,     1.0),
    'B_total'          : (1.0,  100.0,    20.0),
    'D_free'           : (10.0, 500.0,   130.0),
    'alpha'            : (0.001,  1.0,    0.05),
    'dff_offset'       : (-0.5,   0.5,    0.0),
    'K_m'              : (0.5,   10.0,    2.4),
}

# Choose a subset to fit (start with the most impactful parameters):
FIT_PARAMS = [
    'V_basal',
    'V_max_AC',
    'k_cat',
    'PDE_conc',
    'Ca_amplitude',
    'tau_Ca',
    'B_total',
    'alpha',
    'dff_offset',
]

# Compartments to include in the fit, and their weights.
# Upweight γ4-5 to emphasise the spreading phenomenon.
FIT_COMPARTMENTS = ['g1', 'g2', 'g3', 'g4', 'g5']
COMPARTMENT_WEIGHTS = {
    'g1': 1.0,
    'g2': 1.5,   # directly activated
    'g3': 1.0,
    'g4': 2.0,   # key spreading readout
    'g5': 2.0,   # key spreading readout
}

# Stimulation protocol — adjust if you know the exact timing!
# Default: 5 pairings at 30s intervals based on peak analysis
STIM = StimProtocol(
    pairing_onsets=np.array([0.0, 30.0, 60.0, 90.0, 120.0]),
    odor_dur=5.0,
    shock_onset_in_trial=4.0,
    shock_dur=1.0,
    aversive_comps=[0, 1],  # γ1, γ2
)


# =============================================================================
# 5. MAIN OPTIMISATION ROUTINES
# =============================================================================

def run_optimisation(data: Dict, stim: StimProtocol,
                     param_names: List[str] = None,
                     maxiter: int = 200,
                     popsize: int = 20,
                     tol: float = 1e-4,
                     seed: int = 42,
                     n_polish: int = 3,
                     verbose: bool = True,
                     ) -> Tuple[FittableParams, dict]:
    """Run differential evolution + polishing.

    Args:
        data: Preprocessed data (downsampled recommended)
        stim: Stimulation protocol
        param_names: Which parameters to fit
        maxiter: Max DE generations
        popsize: DE population multiplier
        tol: Convergence tolerance
        seed: Random seed
        n_polish: Number of Nelder-Mead restarts for polishing
        verbose: Print progress

    Returns:
        (best_params, result_dict)
    """
    if param_names is None:
        param_names = FIT_PARAMS

    # Build geometry once
    geom = Geometry().build()

    # Set up bounds
    bounds = [PARAM_CONFIG[name][:2] for name in param_names]

    # Base params (defaults for unfitted parameters)
    base = FittableParams()

    # Callback for progress
    iteration = [0]
    best_cost = [1e12]

    def callback(xk, convergence=0):
        iteration[0] += 1
        c = cost_function(xk, data, stim, geom, param_names, base,
                          FIT_COMPARTMENTS, COMPARTMENT_WEIGHTS)
        if c < best_cost[0]:
            best_cost[0] = c
        if verbose and iteration[0] % 10 == 0:
            print(f"  Generation {iteration[0]}: best cost = {best_cost[0]:.4f}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"DIFFERENTIAL EVOLUTION OPTIMISATION")
        print(f"{'='*60}")
        print(f"Fitting {len(param_names)} parameters: {param_names}")
        print(f"Compartments: {FIT_COMPARTMENTS}")
        print(f"Pairings at: {stim.pairing_onsets.tolist()} s")
        print(f"Max generations: {maxiter}, population: {popsize}x")
        print(f"{'='*60}\n")

    t_start = time_module.time()

    result = differential_evolution(
        cost_function,
        bounds=bounds,
        args=(data, stim, geom, param_names, base,
              FIT_COMPARTMENTS, COMPARTMENT_WEIGHTS),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=seed,
        callback=callback,
        disp=verbose,
        workers=1,       # set to -1 for parallel (requires if __name__ guard)
        updating='deferred',
        mutation=(0.5, 1.5),
        recombination=0.8,
    )

    if verbose:
        elapsed = time_module.time() - t_start
        print(f"\nDE completed in {elapsed:.0f}s")
        print(f"  Final cost: {result.fun:.6f}")
        print(f"  Converged: {result.success}")

    # --- Polishing with Nelder-Mead ---
    if verbose:
        print(f"\nPolishing with {n_polish} Nelder-Mead restarts...")

    best_x = result.x.copy()
    best_fun = result.fun

    for i in range(n_polish):
        # Perturb slightly
        x0 = best_x * (1.0 + 0.02 * np.random.randn(len(best_x)))
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

        res = minimize(
            cost_function, x0,
            args=(data, stim, geom, param_names, base,
                  FIT_COMPARTMENTS, COMPARTMENT_WEIGHTS),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6},
        )
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x.copy()
            if verbose:
                print(f"  Polish {i+1}: improved to {best_fun:.6f}")

    # Assemble best parameters
    best_params = FittableParams()
    for name, val in zip(param_names, best_x):
        setattr(best_params, name, val)

    result_dict = {
        'cost': best_fun,
        'param_names': param_names,
        'param_values': best_x.tolist(),
        'all_params': {k: getattr(best_params, k)
                       for k in best_params.__dataclass_fields__
                       if not k.startswith('_')},
        'stim_onsets': stim.pairing_onsets.tolist(),
        'n_generations': result.nit,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"BEST FIT PARAMETERS (cost = {best_fun:.6f})")
        print(f"{'='*60}")
        for name, val in zip(param_names, best_x):
            lo, hi, default = PARAM_CONFIG[name]
            print(f"  {name:20s} = {val:10.4f}  (bounds: [{lo}, {hi}], default: {default})")

    return best_params, result_dict


# =============================================================================
# 6. PLOTTING
# =============================================================================

COMP_COLORS = {
    'g1': '#e41a1c', 'g2': '#ff7f00', 'g3': '#999999',
    'g4': '#377eb8', 'g5': '#4daf4a',
}


def plot_fit(params: FittableParams, data: Dict, stim: StimProtocol,
             save_dir: str = '.', tag: str = '') -> None:
    """Plot model fit vs data for both genotypes.

    Generates a multi-panel figure showing:
      - WT bouton time courses (data + model)
      - dunce-KD bouton time courses (data + model)
      - Residuals summary
    """
    geom = Geometry().build()
    t_model = np.arange(0.0, 180.0, 0.25)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3,
                           height_ratios=[1, 1, 0.6])

    for col, (geno_label, is_dunce, title) in enumerate([
        ('dnc-wt', False, 'Wild-Type'),
        ('dnc-KD', True, 'dunce-KD'),
    ]):
        # Run model
        t_m, bouton_c, _ = run_model(params, stim, is_dunce=is_dunce,
                                      geom=geom, t_eval=t_model)
        pred_dff = model_to_dff(bouton_c, params)

        # --- All compartments overlaid ---
        ax_all = fig.add_subplot(gs[0, col])
        for comp in ['g1', 'g2', 'g3', 'g4', 'g5']:
            ci = COMP_MAP[comp]
            color = COMP_COLORS[comp]
            key = (geno_label, comp)

            # Data
            if key in data:
                t_d, m_d, s_d = data[key]
                mask = t_d >= 0
                ax_all.fill_between(t_d[mask], m_d[mask] - s_d[mask],
                                    m_d[mask] + s_d[mask],
                                    alpha=0.15, color=color)
                ax_all.plot(t_d[mask], m_d[mask], '-', color=color,
                           alpha=0.5, linewidth=0.8)

            # Model
            ax_all.plot(t_m, pred_dff[:, ci], '--', color=color,
                       linewidth=2.0, label=f'{comp} model')

        ax_all.set_title(f'{title} — All Compartments', fontsize=12)
        ax_all.set_ylabel('ΔF/F')
        ax_all.legend(fontsize=7, ncol=3, loc='upper right')
        ax_all.set_xlim(0, 180)

        # Mark pairing times
        for onset in stim.pairing_onsets:
            ax_all.axvline(onset, color='red', alpha=0.15, linewidth=0.5)

        # --- Aversive vs appetitive comparison ---
        ax_comp = fig.add_subplot(gs[1, col])

        # Aversive mean (g1-g2)
        for comp, label, ls in [('g2', 'γ2 (aversive)', '-'),
                                 ('g4', 'γ4 (appetitive)', '-'),
                                 ('g5', 'γ5 (appetitive)', '--')]:
            ci = COMP_MAP[comp]
            color = COMP_COLORS[comp]
            key = (geno_label, comp)

            if key in data:
                t_d, m_d, s_d = data[key]
                mask = t_d >= 0
                ax_comp.fill_between(t_d[mask], m_d[mask] - s_d[mask],
                                     m_d[mask] + s_d[mask],
                                     alpha=0.12, color=color)
                ax_comp.plot(t_d[mask], m_d[mask], ls, color=color,
                            alpha=0.5, linewidth=0.8, label=f'{label} data')

            ax_comp.plot(t_m, pred_dff[:, ci], ls, color=color,
                        linewidth=2, alpha=0.8, label=f'{label} model')

        ax_comp.set_title(f'{title} — Aversive vs Appetitive', fontsize=12)
        ax_comp.set_xlabel('Time (s)')
        ax_comp.set_ylabel('ΔF/F')
        ax_comp.legend(fontsize=7, ncol=2, loc='upper left')
        ax_comp.set_xlim(0, 180)

    # --- Residual bar chart ---
    ax_res = fig.add_subplot(gs[2, :])
    comps = ['g1', 'g2', 'g3', 'g4', 'g5']
    genos = ['dnc-wt', 'dnc-KD']
    x_pos = np.arange(len(comps))
    width = 0.35

    for gi, geno in enumerate(genos):
        rmse_vals = []
        for comp in comps:
            key = (geno, comp)
            if key not in data:
                rmse_vals.append(0)
                continue
            t_d, m_d, _ = data[key]
            mask = t_d >= 0
            ci = COMP_MAP[comp]
            is_dunce = (geno == 'dnc-KD')
            t_m, bc, _ = run_model(params, stim, is_dunce=is_dunce,
                                   geom=geom, t_eval=t_model)
            pred = model_to_dff(bc, params)
            pred_interp = np.interp(t_d[mask], t_m, pred[:, ci])
            rmse = np.sqrt(np.mean((pred_interp - m_d[mask])**2))
            rmse_vals.append(rmse)

        ax_res.bar(x_pos + gi * width, rmse_vals, width,
                   label=geno, alpha=0.7,
                   color=['#2ca02c', '#d62728'][gi])

    ax_res.set_xticks(x_pos + width / 2)
    ax_res.set_xticklabels([f'γ{i+1}' for i in range(5)])
    ax_res.set_ylabel('RMSE (ΔF/F)')
    ax_res.set_title('Fit Quality per Compartment')
    ax_res.legend()

    fig.suptitle('cAMP Nanodomain Model — Fit to Experimental Data',
                 fontsize=14, fontweight='bold')

    path = os.path.join(save_dir, f'fit_result{tag}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\nSaved fit figure to: {path}")
    plt.close(fig)


def plot_parameter_landscape(params: FittableParams, data: Dict,
                             stim: StimProtocol,
                             param_names: List[str] = None,
                             save_dir: str = '.') -> None:
    """1D parameter sweeps around the best fit to show sensitivity."""
    if param_names is None:
        param_names = ['k_cat', 'V_max_AC', 'B_total', 'alpha', 'Ca_amplitude']

    geom = Geometry().build()
    base = FittableParams(**{k: getattr(params, k)
                             for k in params.__dataclass_fields__})

    fig, axes = plt.subplots(1, len(param_names), figsize=(4*len(param_names), 4))
    if len(param_names) == 1:
        axes = [axes]

    for ax, pname in zip(axes, param_names):
        best_val = getattr(params, pname)
        lo, hi, _ = PARAM_CONFIG.get(pname, (best_val*0.1, best_val*3, best_val))

        sweep = np.linspace(lo, hi, 30)
        costs = []
        for v in sweep:
            p = FittableParams(**{k: getattr(base, k)
                                  for k in base.__dataclass_fields__})
            setattr(p, pname, v)
            x_vec = [getattr(p, n) for n in FIT_PARAMS]
            c = cost_function(np.array(x_vec), data, stim, geom, FIT_PARAMS,
                              base, FIT_COMPARTMENTS, COMPARTMENT_WEIGHTS)
            costs.append(c)

        ax.plot(sweep, costs, 'k-', linewidth=1.5)
        ax.axvline(best_val, color='red', linestyle='--', label=f'best={best_val:.3f}')
        ax.set_xlabel(pname)
        ax.set_ylabel('Cost')
        ax.legend(fontsize=8)

    fig.suptitle('Parameter Sensitivity (1D sweeps)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'parameter_sensitivity.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved parameter sensitivity figure")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='cAMP nanodomain model optimisation'
    )
    parser.add_argument('--data', type=str, default='all_camp_long.csv',
                        help='Path to experimental CSV data')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run (fewer iterations)')
    parser.add_argument('--maxiter', type=int, default=200,
                        help='Max DE generations (default: 200)')
    parser.add_argument('--popsize', type=int, default=20,
                        help='DE population multiplier (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for figures and results')
    parser.add_argument('--plot-only', type=str, default=None, metavar='JSON',
                        help='Skip optimisation; plot from saved params JSON')
    parser.add_argument('--stim-onsets', type=str, default=None,
                        help='Comma-separated pairing onset times (s), '
                             'e.g. "0,30,60,90,120"')
    parser.add_argument('--downsample-dt', type=float, default=1.0,
                        help='Downsample data to this dt (s) for fitting')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Stimulation protocol
    stim = StimProtocol()
    if args.stim_onsets:
        stim.pairing_onsets = np.array([float(x) for x in args.stim_onsets.split(',')])
        print(f"Custom stimulation onsets: {stim.pairing_onsets}")

    # Load data
    print(f"Loading data from: {args.data}")
    data_full = load_data(args.data)
    print(f"  Loaded {len(data_full)} traces")

    # Downsample for fitting
    data_ds = downsample_data(data_full, dt_target=args.downsample_dt)
    print(f"  Downsampled to dt={args.downsample_dt}s")

    if args.plot_only:
        # Load saved parameters and plot
        print(f"\nLoading parameters from: {args.plot_only}")
        with open(args.plot_only) as f:
            saved = json.load(f)

        params = FittableParams()
        for k, v in saved['all_params'].items():
            if hasattr(params, k):
                setattr(params, k, v)

        if 'stim_onsets' in saved:
            stim.pairing_onsets = np.array(saved['stim_onsets'])

        plot_fit(params, data_full, stim, save_dir=args.outdir)
        plot_parameter_landscape(params, data_ds, stim, save_dir=args.outdir)
        return

    # Run optimisation
    if args.quick:
        maxiter = 30
        popsize = 10
    else:
        maxiter = args.maxiter
        popsize = args.popsize

    best_params, result_dict = run_optimisation(
        data_ds, stim,
        maxiter=maxiter,
        popsize=popsize,
        seed=args.seed,
    )

    # Save results
    json_path = os.path.join(args.outdir, 'best_params.json')
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved parameters to: {json_path}")

    # Plot using full-resolution data
    plot_fit(best_params, data_full, stim, save_dir=args.outdir)
    plot_parameter_landscape(best_params, data_ds, stim, save_dir=args.outdir)

    print(f"\n{'='*60}")
    print("OPTIMISATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output dir: {args.outdir}")
    print(f"  best_params.json  — fitted parameter values")
    print(f"  fit_result.png    — model vs data comparison")
    print(f"  parameter_sensitivity.png — 1D sweeps")
    print(f"\nTo re-plot with different stim timing:")
    print(f"  python optimise_model.py --plot-only best_params.json "
          f"--stim-onsets '0,30,60,90,120'")


if __name__ == '__main__':
    main()
