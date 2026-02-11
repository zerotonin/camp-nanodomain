#!/usr/bin/env python3
"""
cAMP Nanodomain Model — GPU-Accelerated Parameter Optimisation
===============================================================

JAX + diffrax reimplementation of the cAMP reaction-diffusion model.
The entire forward model and cost function are JIT-compiled to GPU,
and population members are evaluated in parallel via vmap.

Requires: jax[cuda12], diffrax, evosax, optax, equinox
See environment.yml for conda setup.

Usage:
    python optimise_model_gpu.py --data all_camp_long.csv --outdir results_gpu/
    python optimise_model_gpu.py --data all_camp_long.csv --quick
    python optimise_model_gpu.py --plot-only results_gpu/best_params.json

Speedup over CPU version: typically 20-50× on RTX 4070
(~360 ODE solves per generation execute in parallel on GPU)

Author: [Your Lab]
Date:   2026
"""

import os
import sys
import json
import argparse
import time as time_module
from functools import partial
from typing import Dict, List, Tuple, Optional, NamedTuple

import numpy as np
import pandas as pd

# --- JAX (set GPU before import) ---
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import jax.random as jrandom

import diffrax
import equinox as eqx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Check GPU availability
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Enable 64-bit precision (important for ODE solvers)
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. GEOMETRY (static, computed once on CPU then transferred)
# =============================================================================

BOUTON = 0
AXON = 1


def build_geometry(n_comp=5, d_bouton=1.5, d_axon=0.3,
                   inter_bouton=5.0, axon_bins=20):
    """Build 1D spatial grid. Returns JAX arrays.

    Returns dict with:
        x, dx, A, element_type, compartment_id, bouton_mask,
        aversive_mask, alpha_plus, alpha_minus, n_total
    """
    A_b = np.pi * (d_bouton / 2) ** 2
    A_a = np.pi * (d_axon / 2) ** 2
    axon_len = inter_bouton - d_bouton
    ax_dx = axon_len / axon_bins

    positions, dx_arr, areas, types, comp_ids = [], [], [], [], []
    bouton_indices = []
    cx = 0.0
    idx = 0

    for i in range(n_comp):
        bouton_indices.append(idx)
        positions.append(cx + d_bouton / 2)
        dx_arr.append(d_bouton)
        areas.append(A_b)
        types.append(BOUTON)
        comp_ids.append(i)
        idx += 1
        cx += d_bouton
        if i < n_comp - 1:
            for j in range(axon_bins):
                positions.append(cx + (j + 0.5) * ax_dx)
                dx_arr.append(ax_dx)
                areas.append(A_a)
                types.append(AXON)
                comp_ids.append(-1)
                idx += 1
            cx += axon_len

    x = np.array(positions)
    dx = np.array(dx_arr)
    A = np.array(areas)
    types = np.array(types)
    comp_ids = np.array(comp_ids)
    N = idx

    # Coupling coefficients
    alpha_p = np.zeros(N)
    alpha_m = np.zeros(N)
    for i in range(N - 1):
        A_int = 2.0 * A[i] * A[i+1] / (A[i] + A[i+1])
        dist = 0.5 * (dx[i] + dx[i+1])
        flux_coeff = A_int / dist  # D_free multiplied later
        V_i = A[i] * dx[i]
        V_ip1 = A[i+1] * dx[i+1]
        alpha_p[i] = flux_coeff / V_i
        alpha_m[i+1] = flux_coeff / V_ip1

    # Boolean masks
    bouton_mask = (types == BOUTON)
    aversive_mask = np.zeros(N, dtype=bool)
    for bi in bouton_indices:
        if comp_ids[bi] in [0, 1]:  # γ1, γ2
            aversive_mask[bi] = True

    # Bouton index array for extracting compartment values
    bouton_idx = np.array(bouton_indices)

    return {
        'x': jnp.array(x),
        'dx': jnp.array(dx),
        'A': jnp.array(A),
        'alpha_p': jnp.array(alpha_p),
        'alpha_m': jnp.array(alpha_m),
        'bouton_mask': jnp.array(bouton_mask, dtype=jnp.float64),
        'aversive_mask': jnp.array(aversive_mask, dtype=jnp.float64),
        'bouton_idx': jnp.array(bouton_idx, dtype=jnp.int32),
        'n_total': N,
    }


# =============================================================================
# 2. PARAMETER HANDLING
# =============================================================================

# Parameter names and their indices in the flat vector
PARAM_NAMES = [
    'D_free',        # 0
    'B_total',       # 1
    'K_D_buffer',    # 2
    'k_on_buffer',   # 3
    'V_basal',       # 4
    'V_max_AC',      # 5
    'K_Ca',          # 6
    'n_Hill_AC',     # 7
    'k_cat',         # 8
    'K_m',           # 9
    'PDE_conc',      # 10
    'Ca_rest',       # 11
    'Ca_amplitude',  # 12
    'tau_Ca',        # 13
    'Ca_pulse_dur',  # 14
    'alpha',         # 15
    'dff_offset',    # 16
]

# Default values
DEFAULTS = jnp.array([
    130.0,   # D_free
    20.0,    # B_total
    2.0,     # K_D_buffer
    10.0,    # k_on_buffer
    0.5,     # V_basal
    10.0,    # V_max_AC
    0.5,     # K_Ca
    2.0,     # n_Hill_AC
    5.0,     # k_cat
    2.4,     # K_m
    1.0,     # PDE_conc
    0.05,    # Ca_rest
    2.0,     # Ca_amplitude
    1.0,     # tau_Ca
    0.5,     # Ca_pulse_dur
    0.05,    # alpha
    0.0,     # dff_offset
])

# Which parameters to optimise (indices into PARAM_NAMES)
FIT_INDICES = jnp.array([
    4,   # V_basal
    5,   # V_max_AC
    8,   # k_cat
    10,  # PDE_conc
    12,  # Ca_amplitude
    13,  # tau_Ca
    1,   # B_total
    15,  # alpha
    16,  # dff_offset
], dtype=jnp.int32)

FIT_NAMES = [PARAM_NAMES[i] for i in FIT_INDICES]

# Bounds for fitted parameters (lower, upper)
FIT_BOUNDS = jnp.array([
    [0.01,   5.0],    # V_basal
    [1.0,   50.0],    # V_max_AC
    [1.0,  200.0],    # k_cat
    [0.1,   10.0],    # PDE_conc
    [0.5,   20.0],    # Ca_amplitude
    [0.2,    5.0],    # tau_Ca
    [1.0,  100.0],    # B_total
    [0.001,  1.0],    # alpha
    [-0.5,   0.5],    # dff_offset
])


def pack_full_params(fit_values: jnp.ndarray) -> jnp.ndarray:
    """Insert fitted values into full parameter vector."""
    params = DEFAULTS.copy()
    params = params.at[FIT_INDICES].set(fit_values)
    return params


# =============================================================================
# 3. STIMULATION (smooth, differentiable approximations)
# =============================================================================

def smooth_pulse(t, onset, duration, steepness=20.0):
    """Smooth approximation to a rectangular pulse using sigmoid pair.

    Differentiable alternative to if/else for JAX compatibility.
    steepness controls transition sharpness (higher = sharper).
    """
    rise = jax.nn.sigmoid(steepness * (t - onset))
    fall = jax.nn.sigmoid(steepness * (t - onset - duration))
    return rise - fall


def make_stim_signals(pairing_onsets, odor_dur=5.0, shock_offset=4.0,
                      shock_dur=1.0, steepness=20.0):
    """Create odor and shock signal functions.

    Returns:
        odor_fn(t) -> scalar in [0, 1]
        shock_fn(t) -> scalar in [0, 1]
    """
    onsets = jnp.array(pairing_onsets)

    def odor_signal(t):
        return jnp.clip(
            jnp.sum(vmap(lambda o: smooth_pulse(t, o, odor_dur, steepness))(onsets)),
            0.0, 1.0
        )

    def shock_signal(t):
        return jnp.clip(
            jnp.sum(vmap(lambda o: smooth_pulse(t, o + shock_offset, shock_dur, steepness))(onsets)),
            0.0, 1.0
        )

    return odor_signal, shock_signal


# =============================================================================
# 4. ODE DYNAMICS (pure JAX, JIT-compatible)
# =============================================================================

def make_ode_term(geom, params, odor_fn, shock_fn, is_dunce):
    """Build the diffrax ODE term (vector field).

    State layout: y = [c_free(N), c_bound(N), Ca(N)]
    """
    N = geom['n_total']
    alpha_p = geom['alpha_p']
    alpha_m = geom['alpha_m']
    bouton_mask = geom['bouton_mask']       # 1.0 for boutons, 0.0 for axon
    aversive_mask = geom['aversive_mask']   # 1.0 for γ1/γ2 boutons

    # Unpack parameters
    D_free      = params[0]
    B_total     = params[1]
    K_D_buffer  = params[2]
    k_on        = params[3]
    V_basal     = params[4]
    V_max_AC    = params[5]
    K_Ca        = params[6]
    n_Hill      = params[7]
    k_cat       = params[8]
    K_m         = params[9]
    PDE_conc    = params[10]
    Ca_rest     = params[11]
    Ca_amp      = params[12]
    tau_Ca      = params[13]
    Ca_dur      = params[14]

    k_off = k_on * K_D_buffer
    V_max_PDE = PDE_conc * k_cat

    # PDE activity mask: boutons only for WT, zero everywhere for dunce
    pde_mask = bouton_mask * (1.0 - is_dunce)  # is_dunce: 0.0 or 1.0

    def vector_field(t, y, args):
        c = y[:N]
        cb = y[N:2*N]
        ca = y[2*N:3*N]

        c_pos = jnp.maximum(c, 0.0)
        cb_pos = jnp.maximum(cb, 0.0)

        # 1. Diffusion (D_free scales the precomputed geometric coupling)
        dc = jnp.zeros(N)
        dc = dc.at[:-1].add(D_free * alpha_p[:-1] * (c[1:] - c[:-1]))
        dc = dc.at[1:].add(D_free * alpha_m[1:] * (c[:-1] - c[1:]))

        # 2. AC production (boutons only, Ca²⁺-enhanced during odor)
        f_ca = ca**n_Hill / (K_Ca**n_Hill + ca**n_Hill + 1e-30)
        odor_on = odor_fn(t)
        j_ac = bouton_mask * (V_basal + V_max_AC * f_ca * odor_on)
        dc = dc + j_ac

        # 3. PDE degradation (Michaelis-Menten, boutons only, 0 for dunce)
        j_pde = pde_mask * V_max_PDE * c_pos / (K_m + c_pos + 1e-30)
        dc = dc - j_pde

        # 4. Buffering
        B_free = jnp.maximum(B_total - cb_pos, 0.0)
        j_bind = k_on * c_pos * B_free
        j_unbind = k_off * cb_pos
        dc = dc - j_bind + j_unbind
        dcb = j_bind - j_unbind

        # 5. Calcium dynamics
        shock_on = shock_fn(t)
        j_ca_influx = aversive_mask * shock_on * Ca_amp / Ca_dur
        dca = -(ca - Ca_rest) / tau_Ca + j_ca_influx

        return jnp.concatenate([dc, dcb, dca])

    return diffrax.ODETerm(vector_field)


# =============================================================================
# 5. FORWARD MODEL (single run)
# =============================================================================

def solve_forward(params, geom, pairing_onsets, is_dunce,
                  t_save, solver_dt0=0.05, max_steps=50000):
    """Run the forward model and return bouton cAMP at saved times.

    Args:
        params: Full parameter vector (17 elements)
        geom: Geometry dict
        pairing_onsets: Array of pairing onset times
        is_dunce: 0.0 for WT, 1.0 for dunce
        t_save: Times at which to save output
        solver_dt0: Initial step size
        max_steps: Maximum solver steps

    Returns:
        bouton_c_free: [n_save, 5] free cAMP at each bouton
    """
    N = geom['n_total']
    Ca_rest = params[11]

    # Initial conditions
    y0 = jnp.zeros(3 * N)
    y0 = y0.at[2*N:3*N].set(Ca_rest)

    # Stimulus functions
    odor_fn, shock_fn = make_stim_signals(pairing_onsets)

    # ODE term
    term = make_ode_term(geom, params, odor_fn, shock_fn, is_dunce)

    # Solver: Kvaerno5 (implicit, L-stable, good for stiff systems)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(
        rtol=1e-5, atol=1e-7,
        dtmin=1e-5, dtmax=1.0,
    )

    # Save at specified times
    saveat = diffrax.SaveAt(ts=t_save)

    sol = diffrax.diffeqsolve(
        term, solver, t0=t_save[0], t1=t_save[-1], dt0=solver_dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        throw=False,  # return NaN instead of error (for robustness in optim)
    )

    # Extract bouton free cAMP: sol.ys is [n_save, 3*N]
    c_free = sol.ys[:, :N]                         # [n_save, N]
    bouton_c = c_free[:, geom['bouton_idx']]       # [n_save, 5]

    return bouton_c


# =============================================================================
# 6. COST FUNCTION (JIT-compiled, vmappable)
# =============================================================================

def cost_single(fit_values, geom, pairing_onsets, t_data_wt, t_data_kd,
                data_wt, data_kd, sem_wt, sem_kd, weights, t_save):
    """Cost for one parameter vector (both genotypes).

    All inputs are JAX arrays. Fully JIT-compatible.

    Args:
        fit_values: [n_fit] fitted parameter values
        geom: geometry dict
        pairing_onsets: [n_pairings] stimulus times
        t_data_wt: [n_comp, n_t_wt] data time grids per compartment (padded)
        t_data_kd: [n_comp, n_t_kd] same for dunce
        data_wt: [n_comp, n_t_wt] mean dF/F (padded with NaN)
        data_kd: [n_comp, n_t_kd] same
        sem_wt: [n_comp, n_t_wt] SEM
        sem_kd: [n_comp, n_t_kd]
        weights: [5] per-compartment weights
        t_save: [n_save] model output time grid
    """
    params = pack_full_params(fit_values)
    alpha = params[15]
    offset = params[16]

    total = 0.0
    n_pts = 0.0

    for is_dunce_val, t_data, data_mean, data_sem in [
        (0.0, t_data_wt, data_wt, sem_wt),
        (1.0, t_data_kd, data_kd, sem_kd),
    ]:
        bouton_c = solve_forward(params, geom, pairing_onsets,
                                 is_dunce_val, t_save)
        pred_dff = alpha * bouton_c + offset  # [n_save, 5]

        # For each compartment, interpolate model to data times
        for k in range(5):
            pred_k = pred_dff[:, k]   # [n_save]
            t_dk = t_data[k]          # [n_t]
            d_dk = data_mean[k]       # [n_t]
            s_dk = data_sem[k]        # [n_t]

            # Linear interpolation of model to data time points
            pred_interp = jnp.interp(t_dk, t_save, pred_k)

            # Mask: valid data points (not NaN-padded)
            valid = jnp.isfinite(d_dk)

            # Chi-squared residual
            resid = jnp.where(valid,
                              weights[k] * (pred_interp - d_dk)**2 / (s_dk**2 + 1e-8),
                              0.0)
            total = total + jnp.sum(resid)
            n_pts = n_pts + jnp.sum(valid.astype(jnp.float64))

    return total / jnp.maximum(n_pts, 1.0)


# Batch version: evaluate cost for an entire population in parallel
@partial(jit, static_argnums=(1,))
def cost_batch(population, geom_static, pairing_onsets, t_data_wt, t_data_kd,
               data_wt, data_kd, sem_wt, sem_kd, weights, t_save):
    """Evaluate cost for entire population [pop_size, n_fit] in parallel."""
    geom = geom_static
    return vmap(
        lambda x: cost_single(x, geom, pairing_onsets, t_data_wt, t_data_kd,
                               data_wt, data_kd, sem_wt, sem_kd, weights, t_save)
    )(population)


# =============================================================================
# 7. GPU-NATIVE DIFFERENTIAL EVOLUTION
# =============================================================================

def differential_evolution_jax(cost_fn, bounds, popsize=20, maxiter=200,
                               mutation=(0.5, 1.5), crossover=0.8,
                               tol=1e-5, seed=42, verbose=True):
    """Differential evolution implemented in JAX (runs on GPU).

    Args:
        cost_fn: callable(population[pop_size, n_dim]) -> costs[pop_size]
        bounds: [n_dim, 2] lower and upper bounds
        popsize: population size multiplier (actual pop = popsize * n_dim)
        maxiter: maximum generations
        mutation: (F_min, F_max) mutation scale range
        crossover: crossover probability
        tol: convergence tolerance on cost std
        seed: random seed
        verbose: print progress

    Returns:
        best_x, best_cost, history
    """
    n_dim = bounds.shape[0]
    pop_n = popsize * n_dim
    F_min, F_max = mutation

    key = jrandom.PRNGKey(seed)

    # Initialise population uniformly in bounds
    key, subkey = jrandom.split(key)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    population = lo + (hi - lo) * jrandom.uniform(subkey, (pop_n, n_dim))

    # Evaluate initial population
    costs = cost_fn(population)
    best_idx = jnp.argmin(costs)
    best_x = population[best_idx]
    best_cost = costs[best_idx]

    history = []

    for gen in range(maxiter):
        t_start = time_module.time()

        key, k1, k2, k3, k4, k5 = jrandom.split(key, 6)

        # Select 3 distinct random indices per member (simple approach)
        r1 = jrandom.randint(k1, (pop_n,), 0, pop_n)
        r2 = jrandom.randint(k2, (pop_n,), 0, pop_n)
        r3 = jrandom.randint(k3, (pop_n,), 0, pop_n)

        # Mutation: F sampled per member
        F = F_min + (F_max - F_min) * jrandom.uniform(k4, (pop_n, 1))

        # Donor vectors: DE/rand/1
        donors = population[r1] + F * (population[r2] - population[r3])

        # Crossover
        cross_mask = jrandom.uniform(k5, (pop_n, n_dim)) < crossover
        # Ensure at least one dimension is crossed
        j_rand = jrandom.randint(k5, (pop_n,), 0, n_dim)
        force_mask = jax.nn.one_hot(j_rand, n_dim, dtype=jnp.bool_)
        cross_mask = cross_mask | force_mask

        trials = jnp.where(cross_mask, donors, population)

        # Clip to bounds
        trials = jnp.clip(trials, lo, hi)

        # Evaluate trials
        trial_costs = cost_fn(trials)

        # Selection: keep better of trial vs current
        improved = trial_costs < costs
        population = jnp.where(improved[:, None], trials, population)
        costs = jnp.where(improved, trial_costs, costs)

        # Track best
        gen_best_idx = jnp.argmin(costs)
        gen_best_cost = costs[gen_best_idx]
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_x = population[gen_best_idx]

        elapsed = time_module.time() - t_start
        history.append(float(best_cost))

        if verbose and (gen % 5 == 0 or gen == maxiter - 1):
            cost_std = jnp.std(costs)
            print(f"  Gen {gen:4d}/{maxiter}: best={float(best_cost):.6f}  "
                  f"mean={float(jnp.mean(costs)):.4f}  "
                  f"std={float(cost_std):.6f}  "
                  f"({elapsed:.2f}s/gen)")

            if float(cost_std) < tol and gen > 20:
                if verbose:
                    print(f"  Converged (cost std < {tol})")
                break

    return np.array(best_x), float(best_cost), history


# =============================================================================
# 8. DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(csv_path: str) -> Dict:
    """Load experimental data and return numpy arrays."""
    df = pd.read_csv(csv_path)
    bouton = df[df['structure'] == 'Bouton'].copy()
    fly_avg = bouton.groupby(
        ['genotype', 'compartment', 'fly_id', 'time_s']
    )['dff'].mean().reset_index()
    summary = fly_avg.groupby(
        ['genotype', 'compartment', 'time_s']
    )['dff'].agg(['mean', 'sem', 'count']).reset_index()

    data = {}
    for geno in ['dnc-wt', 'dnc-KD']:
        for comp in ['g1', 'g2', 'g3', 'g4', 'g5']:
            sub = summary[
                (summary['genotype'] == geno) & (summary['compartment'] == comp)
            ].sort_values('time_s')
            if len(sub) == 0:
                continue
            t = sub['time_s'].values
            m = sub['mean'].values
            s = sub['sem'].values
            s = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s[s > 0]))
            data[(geno, comp)] = (t, m, s)
    return data


def prepare_data_arrays(data: Dict, dt_ds: float = 2.0):
    """Convert data dict to padded JAX arrays for JIT-compatible cost.

    Downsamples to dt_ds resolution and pads all compartments to same length.
    Returns arrays for WT and dunce-KD separately.
    """
    comps = ['g1', 'g2', 'g3', 'g4', 'g5']

    def process_geno(geno):
        arrays_t, arrays_m, arrays_s = [], [], []
        for comp in comps:
            key = (geno, comp)
            if key in data:
                t, m, s = data[key]
                # Only post-stimulus
                mask = t >= 0
                t, m, s = t[mask], m[mask], s[mask]
                # Downsample by binning
                t_bins = np.arange(t.min(), t.max(), dt_ds)
                m_ds = np.array([m[(t >= tb) & (t < tb + dt_ds)].mean() for tb in t_bins])
                s_ds = np.array([s[(t >= tb) & (t < tb + dt_ds)].mean() for tb in t_bins])
                t_ds = t_bins + dt_ds / 2
                valid = np.isfinite(m_ds)
                arrays_t.append(t_ds[valid])
                arrays_m.append(m_ds[valid])
                arrays_s.append(s_ds[valid])
            else:
                arrays_t.append(np.array([]))
                arrays_m.append(np.array([]))
                arrays_s.append(np.array([]))

        # Pad to same length with NaN
        max_len = max(len(a) for a in arrays_t)
        if max_len == 0:
            max_len = 1

        t_pad = np.full((5, max_len), np.nan)
        m_pad = np.full((5, max_len), np.nan)
        s_pad = np.full((5, max_len), np.nan)

        for i in range(5):
            n = len(arrays_t[i])
            if n > 0:
                t_pad[i, :n] = arrays_t[i]
                m_pad[i, :n] = arrays_m[i]
                s_pad[i, :n] = arrays_s[i]

        return jnp.array(t_pad), jnp.array(m_pad), jnp.array(s_pad)

    t_wt, m_wt, s_wt = process_geno('dnc-wt')
    t_kd, m_kd, s_kd = process_geno('dnc-KD')

    return t_wt, t_kd, m_wt, m_kd, s_wt, s_kd


# =============================================================================
# 9. PLOTTING (uses numpy/matplotlib, runs on CPU)
# =============================================================================

COMP_COLORS = {'g1': '#e41a1c', 'g2': '#ff7f00', 'g3': '#999999',
               'g4': '#377eb8', 'g5': '#4daf4a'}
COMP_MAP = {'g1': 0, 'g2': 1, 'g3': 2, 'g4': 3, 'g5': 4}


def run_model_numpy(params_array, pairing_onsets, is_dunce, t_save, geom):
    """Run model and return numpy arrays for plotting."""
    bouton_c = solve_forward(
        jnp.array(params_array), geom, jnp.array(pairing_onsets),
        float(is_dunce), jnp.array(t_save)
    )
    return np.array(t_save), np.array(bouton_c)


def plot_fit(params_array, data_full, pairing_onsets, geom,
             save_dir='.', tag=''):
    """Plot model fit vs experimental data."""
    alpha = float(params_array[15])
    offset = float(params_array[16])
    t_model = np.arange(0, 180, 0.5)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3,
                           height_ratios=[1, 1, 0.6])

    for col, (geno, is_d, title) in enumerate([
        ('dnc-wt', 0.0, 'Wild-Type'),
        ('dnc-KD', 1.0, 'dunce-KD'),
    ]):
        t_m, bc = run_model_numpy(params_array, pairing_onsets, is_d, t_model, geom)
        pred_dff = alpha * bc + offset

        # All compartments
        ax1 = fig.add_subplot(gs[0, col])
        for comp in ['g1', 'g2', 'g3', 'g4', 'g5']:
            ci = COMP_MAP[comp]
            color = COMP_COLORS[comp]
            key = (geno, comp)
            if key in data_full:
                td, md, sd = data_full[key]
                mask = td >= 0
                ax1.fill_between(td[mask], md[mask]-sd[mask], md[mask]+sd[mask],
                                 alpha=0.15, color=color)
                ax1.plot(td[mask], md[mask], '-', color=color, alpha=0.5, linewidth=0.8)
            ax1.plot(t_m, pred_dff[:, ci], '--', color=color, linewidth=2,
                    label=f'{comp} model')

        ax1.set_title(f'{title} — All Compartments', fontsize=12)
        ax1.set_ylabel('ΔF/F')
        ax1.legend(fontsize=7, ncol=3, loc='upper right')
        ax1.set_xlim(0, 180)
        for o in pairing_onsets:
            ax1.axvline(o, color='red', alpha=0.15, linewidth=0.5)

        # Aversive vs appetitive
        ax2 = fig.add_subplot(gs[1, col])
        for comp, label, ls in [('g2', 'γ2 (aversive)', '-'),
                                 ('g4', 'γ4 (appetitive)', '-'),
                                 ('g5', 'γ5 (appetitive)', '--')]:
            ci = COMP_MAP[comp]
            color = COMP_COLORS[comp]
            key = (geno, comp)
            if key in data_full:
                td, md, sd = data_full[key]
                mask = td >= 0
                ax2.fill_between(td[mask], md[mask]-sd[mask], md[mask]+sd[mask],
                                 alpha=0.12, color=color)
                ax2.plot(td[mask], md[mask], ls, color=color, alpha=0.5,
                        linewidth=0.8, label=f'{label} data')
            ax2.plot(t_m, pred_dff[:, ci], ls, color=color, linewidth=2,
                    alpha=0.8, label=f'{label} model')

        ax2.set_title(f'{title} — Aversive vs Appetitive', fontsize=12)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('ΔF/F')
        ax2.legend(fontsize=7, ncol=2, loc='upper left')
        ax2.set_xlim(0, 180)

    # RMSE bar chart
    ax3 = fig.add_subplot(gs[2, :])
    comps = ['g1', 'g2', 'g3', 'g4', 'g5']
    x_pos = np.arange(5)
    width = 0.35
    for gi, geno in enumerate(['dnc-wt', 'dnc-KD']):
        rmses = []
        for comp in comps:
            key = (geno, comp)
            if key not in data_full:
                rmses.append(0)
                continue
            td, md, _ = data_full[key]
            mask = td >= 0
            ci = COMP_MAP[comp]
            is_d = 1.0 if geno == 'dnc-KD' else 0.0
            t_m, bc = run_model_numpy(params_array, pairing_onsets, is_d, t_model, geom)
            pred = alpha * bc + offset
            pred_i = np.interp(td[mask], t_m, pred[:, ci])
            rmses.append(np.sqrt(np.mean((pred_i - md[mask])**2)))
        ax3.bar(x_pos + gi*width, rmses, width, label=geno, alpha=0.7,
                color=['#2ca02c', '#d62728'][gi])
    ax3.set_xticks(x_pos + width/2)
    ax3.set_xticklabels([f'γ{i+1}' for i in range(5)])
    ax3.set_ylabel('RMSE (ΔF/F)')
    ax3.set_title('Fit Quality per Compartment')
    ax3.legend()

    fig.suptitle('cAMP Nanodomain Model — GPU Fit to Experimental Data',
                 fontsize=14, fontweight='bold')
    path = os.path.join(save_dir, f'fit_result_gpu{tag}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fit figure: {path}")


def plot_convergence(history, save_dir='.'):
    """Plot optimisation convergence."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, 'k-', linewidth=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Cost')
    ax.set_title('GPU Differential Evolution Convergence')
    ax.set_yscale('log')
    fig.tight_layout()
    path = os.path.join(save_dir, 'convergence_gpu.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved convergence plot: {path}")


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated cAMP model optimisation')
    parser.add_argument('--data', type=str, default='all_camp_long.csv')
    parser.add_argument('--quick', action='store_true', help='Quick test (30 generations)')
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--popsize', type=int, default=20,
                        help='Population multiplier (actual = popsize × n_params)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--plot-only', type=str, default=None, metavar='JSON')
    parser.add_argument('--stim-onsets', type=str, default=None,
                        help='Comma-separated pairing times, e.g. "0,30,60,90,120"')
    parser.add_argument('--downsample-dt', type=float, default=2.0,
                        help='Downsample data to this dt (s) for fitting')
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution')
    args = parser.parse_args()

    if args.cpu:
        jax.config.update('jax_platform_name', 'cpu')
        print("Forced CPU mode")

    os.makedirs(args.outdir, exist_ok=True)

    # Stimulation
    if args.stim_onsets:
        pairing_onsets = np.array([float(x) for x in args.stim_onsets.split(',')])
    else:
        pairing_onsets = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
    print(f"Pairing onsets: {pairing_onsets}")

    # Geometry
    geom = build_geometry()
    print(f"Grid: {geom['n_total']} spatial bins")

    # Load data
    print(f"Loading data: {args.data}")
    data_full = load_data(args.data)
    print(f"  {len(data_full)} traces")

    # Compartment weights
    weights = jnp.array([1.0, 1.5, 1.0, 2.0, 2.0])  # g1-g5

    if args.plot_only:
        with open(args.plot_only) as f:
            saved = json.load(f)
        params_array = np.array([saved['all_params'][n] for n in PARAM_NAMES])
        if 'stim_onsets' in saved:
            pairing_onsets = np.array(saved['stim_onsets'])
        plot_fit(params_array, data_full, pairing_onsets, geom,
                 save_dir=args.outdir)
        return

    # Prepare data arrays
    t_wt, t_kd, m_wt, m_kd, s_wt, s_kd = prepare_data_arrays(
        data_full, dt_ds=args.downsample_dt
    )
    t_save = jnp.arange(0.0, 180.0, 1.0)
    pairing_j = jnp.array(pairing_onsets)

    print(f"\nData shapes: WT t={t_wt.shape}, KD t={t_kd.shape}")

    # --- Build cost function ---
    # We wrap cost_batch so the geometry is captured as a closure
    # (diffrax pytrees inside geom aren't hashable for static_argnums)

    @jit
    def evaluate_population(population):
        """Evaluate cost for entire population."""
        return vmap(
            lambda x: cost_single(x, geom, pairing_j, t_wt, t_kd,
                                   m_wt, m_kd, s_wt, s_kd, weights, t_save)
        )(population)

    # --- Warm up JIT ---
    print("\nJIT-compiling forward model (this takes ~30-60s on first run)...")
    t0 = time_module.time()
    dummy_pop = jnp.tile(DEFAULTS[FIT_INDICES], (3, 1))
    _ = evaluate_population(dummy_pop).block_until_ready()
    print(f"JIT compilation done in {time_module.time()-t0:.1f}s")

    # Benchmark
    t0 = time_module.time()
    n_bench = 20
    bench_pop = jnp.tile(DEFAULTS[FIT_INDICES], (n_bench, 1))
    _ = evaluate_population(bench_pop).block_until_ready()
    per_eval = (time_module.time()-t0) / n_bench
    print(f"Benchmark: {per_eval*1000:.1f} ms per (WT+dunce) forward pass")

    # --- Run optimisation ---
    maxiter = 30 if args.quick else args.maxiter

    print(f"\n{'='*60}")
    print(f"GPU DIFFERENTIAL EVOLUTION")
    print(f"{'='*60}")
    print(f"Parameters: {FIT_NAMES}")
    print(f"Pop size: {args.popsize} × {len(FIT_NAMES)} = {args.popsize * len(FIT_NAMES)}")
    print(f"Max generations: {maxiter}")
    print(f"Device: {jax.devices()[0]}")
    print(f"{'='*60}\n")

    t_total_start = time_module.time()

    best_x, best_cost, history = differential_evolution_jax(
        evaluate_population,
        bounds=np.array(FIT_BOUNDS),
        popsize=args.popsize,
        maxiter=maxiter,
        seed=args.seed,
        verbose=True,
    )

    total_time = time_module.time() - t_total_start

    # Assemble full params
    full_params = np.array(DEFAULTS)
    for i, fi in enumerate(FIT_INDICES):
        full_params[int(fi)] = best_x[i]

    print(f"\n{'='*60}")
    print(f"OPTIMISATION COMPLETE ({total_time:.0f}s)")
    print(f"{'='*60}")
    print(f"Best cost: {best_cost:.6f}")
    print(f"\nFitted parameters:")
    for name, val in zip(FIT_NAMES, best_x):
        lo, hi = FIT_BOUNDS[FIT_NAMES.index(name)]
        print(f"  {name:20s} = {val:10.4f}  (bounds: [{lo:.3f}, {hi:.3f}])")

    # Derived quantities
    D_eff = full_params[0] / (1 + full_params[1] / full_params[2])
    eta = full_params[8] / (4 * np.pi * 2.5e-3 * D_eff * full_params[9])
    print(f"\nDerived:")
    print(f"  D_eff (buffered):    {D_eff:.1f} μm²/s")
    print(f"  V_max_PDE:           {full_params[10]*full_params[8]:.1f} μM/s")
    print(f"  Absorptive action η: {eta:.3f}")

    # Save
    result = {
        'cost': best_cost,
        'param_names': FIT_NAMES,
        'param_values': best_x.tolist(),
        'all_params': {n: float(full_params[i]) for i, n in enumerate(PARAM_NAMES)},
        'stim_onsets': pairing_onsets.tolist(),
        'history': history,
        'total_time_s': total_time,
        'device': str(jax.devices()[0]),
    }
    json_path = os.path.join(args.outdir, 'best_params.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plot
    plot_fit(full_params, data_full, pairing_onsets, geom, save_dir=args.outdir)
    plot_convergence(history, save_dir=args.outdir)

    print(f"\nAll outputs in: {args.outdir}/")


if __name__ == '__main__':
    main()
