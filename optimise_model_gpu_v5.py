#!/usr/bin/env python3
"""
cAMP Nanodomain Model — GPU-Accelerated Parameter Optimisation v5
==================================================================

Updates from v4:
  - K_Ca is now a FREE parameter with bounds [0.03, 0.5] μM.
    Root cause of v3/v4 optimizer failure: K_Ca = 0.5 μM (fixed) sits
    far above free [Ca²⁺] achievable after RBA buffering (~50-200 nM),
    making the Hill function f(Ca) ≈ 0. The optimizer shut off the entire
    stimulus pathway and compensated with V_basal ≈ 4 μM/s.

    Biophysical justification: the effective K_Ca for rutabaga reflects
    competition between calmodulin and endogenous Ca²⁺ buffers (κ_S=77).
    The apparent affinity in the presence of competing buffers is lower
    than the intrinsic CaM-rutabaga K_D.

  - All v4 physics retained:
      * RBA: κ_S = 77, τ_Ca = 46 ms (Bhatt & Bhatt 2011)
      * Dual Ca²⁺: odour (sustained) + shock (transient)
      * Two-term AC: V_Ca (all boutons) + V_coinc (DA-gated, γ1-γ3)
      * DA mask: [1.0, 1.0, 0.5, 0.0, 0.0]

  - Parameter count: 12 free (was 11)
      Added: K_Ca [0.03, 0.5] μM

Usage:
    python optimise_model_gpu_v5.py --data all_camp_long.csv --outdir results_v5/
    python optimise_model_gpu_v5.py --data all_camp_long.csv --quick
    python optimise_model_gpu_v5.py --plot-only results_v5/best_params.json
"""

import os
import sys
import json
import argparse
import time as time_module
from functools import partial
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

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

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. GEOMETRY
# =============================================================================

BOUTON = 0
AXON = 1


def build_geometry(n_comp=5, d_bouton=2.0, d_axon=0.3,
                   inter_comp=25.0, axon_bins=40):
    """Build 1D spatial grid for a single KC axon through the γ-lobe.

    Realistic Drosophila mushroom body γ-lobe dimensions:
      - γ-lobe total length: ~100-125 μm (Aso et al. 2014, Scheffer et al. 2020)
      - Each compartment (γ1-γ5): ~20-25 μm
      - KC axon diameter: ~0.2-0.5 μm
      - Boutons: ~1-2 μm diameter, 3-8 per compartment per KC
    """
    A_b = np.pi * (d_bouton / 2) ** 2
    A_a = np.pi * (d_axon / 2) ** 2
    axon_len = inter_comp - d_bouton
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
    types_arr = np.array(types)
    N = idx

    alpha_p = np.zeros(N)
    alpha_m = np.zeros(N)
    for i in range(N - 1):
        A_int = 2.0 * A[i] * A[i+1] / (A[i] + A[i+1])
        dist = 0.5 * (dx[i] + dx[i+1])
        flux_coeff = A_int / dist
        V_i = A[i] * dx[i]
        V_ip1 = A[i+1] * dx[i+1]
        alpha_p[i] = flux_coeff / V_i
        alpha_m[i+1] = flux_coeff / V_ip1

    bouton_mask = (types_arr == BOUTON).astype(float)
    stim_mask = bouton_mask.copy()

    # DA mask: compartment-specific DAN innervation during aversive shock
    da_mask_per_comp = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
    da_mask = np.zeros(N)
    for i, bi in enumerate(bouton_indices):
        da_mask[bi] = da_mask_per_comp[i]

    return {
        'x': jnp.array(x),
        'dx': jnp.array(dx),
        'A': jnp.array(A),
        'alpha_p': jnp.array(alpha_p),
        'alpha_m': jnp.array(alpha_m),
        'bouton_mask': jnp.array(bouton_mask),
        'stim_mask': jnp.array(stim_mask),
        'da_mask': jnp.array(da_mask),
        'bouton_idx': jnp.array(bouton_indices, dtype=jnp.int32),
        'n_total': N,
    }


# =============================================================================
# 2. PARAMETERS
# =============================================================================

PARAM_NAMES = [
    'D_free',        # 0   μm²/s
    'B_total',       # 1   μM          cAMP buffer (PKA-R)
    'K_D_buffer',    # 2   μM          cAMP buffer K_D
    'k_on_buffer',   # 3   μM⁻¹s⁻¹    cAMP buffer on-rate
    'V_basal',       # 4   μM/s        basal AC rate
    'V_Ca',          # 5   μM/s        Ca²⁺-only rutabaga rate
    'K_Ca',          # 6   μM          half-max Ca²⁺ for AC
    'n_Hill_AC',     # 7               Hill coefficient
    'k_cat',         # 8   s⁻¹         PDE turnover
    'K_m',           # 9   μM          PDE Michaelis constant
    'PDE_conc',      # 10  μM          local [PDE]
    'Ca_rest',       # 11  μM          resting free [Ca²⁺]
    'Ca_odor',       # 12  μM          total Ca²⁺ influx rate during odour
    'tau_Ca',        # 13  s           free Ca²⁺ decay (FIXED from literature)
    'kappa_S',       # 14              endogenous Ca²⁺ binding ratio (FIXED)
    'alpha',         # 15  ΔF/F per μM sensor gain
    'dff_offset',    # 16              baseline offset
    'tau_sensor',    # 17  s           G-Flamp1 response time
    'V_coinc',       # 18  μM/s        coincidence (Ca²⁺+DA) rate
    'Ca_shock',      # 19  μM          total Ca²⁺ influx rate during shock
]

DEFAULTS = jnp.array([
    130.0,   # D_free          Bhatt et al. 2024
    20.0,    # B_total
    2.0,     # K_D_buffer
    10.0,    # k_on_buffer
    0.5,     # V_basal
    2.0,     # V_Ca
    0.5,     # K_Ca
    2.0,     # n_Hill_AC
    5.0,     # k_cat
    2.4,     # K_m             Byers et al. 1981
    1.0,     # PDE_conc
    0.05,    # Ca_rest         ~50 nM resting free [Ca²⁺]
    5.0,     # Ca_odor         total influx (before RBA scaling)
    0.046,   # tau_Ca          FIXED: 46 ms, Bhatt & Bhatt 2011
    77.0,    # kappa_S         FIXED: κ_S = 77, Bhatt & Bhatt 2011
    0.1,     # alpha
    0.0,     # dff_offset
    3.0,     # tau_sensor
    15.0,    # V_coinc
    20.0,    # Ca_shock        total influx during shock (before RBA)
])

# --- Which parameters to fit and their bounds ---
# 12 free parameters (was 11 in v4)
# Added: K_Ca — now free to match RBA-buffered free [Ca²⁺] range
FIT_INDICES = jnp.array([
    4,   # V_basal
    5,   # V_Ca
    18,  # V_coinc
    8,   # k_cat
    10,  # PDE_conc
    6,   # K_Ca          ← NEW: free parameter
    12,  # Ca_odor
    19,  # Ca_shock
    1,   # B_total
    15,  # alpha
    16,  # dff_offset
    17,  # tau_sensor
], dtype=jnp.int32)

FIT_NAMES = [PARAM_NAMES[int(i)] for i in FIT_INDICES]

FIT_BOUNDS = jnp.array([
    [0.01,    5.0],    # V_basal       basal AC production
    [0.1,    20.0],    # V_Ca          Ca²⁺-only (small but detectable)
    [1.0,   200.0],    # V_coinc       coincidence (much larger)
    [0.5,   200.0],    # k_cat         PDE turnover
    [0.1,    10.0],    # PDE_conc      local enzyme concentration
    [0.03,    0.5],    # K_Ca          ← NEW: half-max Ca²⁺ for AC
                       #   Lower bound 30 nM: near resting free [Ca²⁺]
                       #   Upper bound 500 nM: intrinsic CaM K_D
                       #   With RBA, free [Ca²⁺] peaks at ~50-300 nM,
                       #   so K_Ca must be in this range for f(Ca) > 0
    [0.5,    50.0],    # Ca_odor       total Ca²⁺ rate during odour
    [1.0,   200.0],    # Ca_shock      total Ca²⁺ rate during shock
    [1.0,   100.0],    # B_total       cAMP buffer
    [0.001,   2.0],    # alpha         sensor gain
    [-0.5,    0.5],    # dff_offset    baseline
    [0.5,    10.0],    # tau_sensor    G-Flamp1 (tightened)
])


def pack_full_params(fit_values: jnp.ndarray) -> jnp.ndarray:
    """Insert fitted values into full parameter vector."""
    params = DEFAULTS.copy()
    params = params.at[FIT_INDICES].set(fit_values)
    return params


# =============================================================================
# 3. STIMULATION (smooth pulses for JAX compatibility)
# =============================================================================

def smooth_pulse(t, onset, duration, steepness=20.0):
    rise = jax.nn.sigmoid(steepness * (t - onset))
    fall = jax.nn.sigmoid(steepness * (t - onset - duration))
    return rise - fall


def make_stim_signals(pairing_onsets, odor_dur=5.0, shock_offset=4.0,
                      shock_dur=1.0, steepness=20.0):
    """Create odour and shock timing signals.

    Each pairing:
      t=0:             odour ON → KC depolarisation → Ca²⁺ influx (all boutons)
      t=shock_offset:  shock → DANs fire → DA release (γ1-γ2 via DA mask)
                       also → additional Ca²⁺ transient (circuit reverb)
      t=odor_dur:      odour OFF → Ca²⁺ influx stops
    """
    onsets = jnp.array(pairing_onsets)

    def odor_signal(t):
        return jnp.clip(
            jnp.sum(vmap(lambda o: smooth_pulse(t, o, odor_dur, steepness))(onsets)),
            0.0, 1.0)

    def shock_signal(t):
        return jnp.clip(
            jnp.sum(vmap(lambda o: smooth_pulse(t, o + shock_offset, shock_dur, steepness))(onsets)),
            0.0, 1.0)

    return odor_signal, shock_signal


# =============================================================================
# 4. ODE DYNAMICS
# =============================================================================

def make_ode_term(geom, params, odor_fn, shock_fn, is_dunce):
    """Build ODE right-hand side.

    State: y = [c_free(N), c_bound(N), Ca_free(N)]

    Key physics:
      - Ca²⁺ uses Rapid Buffering Approximation (RBA):
          dCa_free/dt = -(Ca - Ca_rest)/τ_Ca + J_total/(1 + κ_S)
        where κ_S = 77 means only 1.3% of entering Ca²⁺ is free.
        τ_Ca = 46 ms gives fast decay between shocks.

      - AC production has two terms:
          1. V_Ca × f(Ca) × odor_on     [all boutons, small]
          2. V_coinc × f(Ca) × DA × shock_on  [γ1-γ3, supralinear]

      - cAMP buffering is explicit (PKA-R subunits).
    """
    N = geom['n_total']
    alpha_p = geom['alpha_p']
    alpha_m = geom['alpha_m']
    bouton_mask = geom['bouton_mask']
    stim_mask = geom['stim_mask']
    da_mask = geom['da_mask']

    D_free      = params[0]
    B_total     = params[1]
    K_D_buffer  = params[2]
    k_on        = params[3]
    V_basal     = params[4]
    V_Ca        = params[5]
    K_Ca        = params[6]
    n_Hill      = params[7]
    k_cat       = params[8]
    K_m         = params[9]
    PDE_conc    = params[10]
    Ca_rest     = params[11]
    Ca_odor     = params[12]     # total Ca²⁺ influx rate during odour
    tau_Ca      = params[13]     # FIXED: 0.046 s
    kappa_S     = params[14]     # FIXED: 77
    V_coinc     = params[18]
    Ca_shock    = params[19]     # total Ca²⁺ influx rate during shock

    k_off = k_on * K_D_buffer
    V_max_PDE = PDE_conc * k_cat
    pde_mask = bouton_mask * (1.0 - is_dunce)

    # RBA scaling factor: fraction of influxing Ca²⁺ that remains free
    rba_factor = 1.0 / (1.0 + kappa_S)   # ≈ 0.013 for κ_S = 77

    def vector_field(t, y, args):
        c  = y[:N]
        cb = y[N:2*N]
        ca = y[2*N:3*N]        # FREE [Ca²⁺] (after buffering)

        c_pos  = jnp.maximum(c, 0.0)
        cb_pos = jnp.maximum(cb, 0.0)

        # ---- cAMP Diffusion ----
        dc = jnp.zeros(N)
        dc = dc.at[:-1].add(D_free * alpha_p[:-1] * (c[1:] - c[:-1]))
        dc = dc.at[1:].add(D_free * alpha_m[1:] * (c[:-1] - c[1:]))

        # ---- AC production (two-term model) ----
        f_ca = ca**n_Hill / (K_Ca**n_Hill + ca**n_Hill + 1e-30)

        odor_on = odor_fn(t)
        shock_on = shock_fn(t)

        # Term 1: Ca²⁺-only rutabaga (all boutons during odour)
        j_ac_ca = bouton_mask * V_Ca * f_ca * odor_on

        # Term 2: Coincidence detection (Ca²⁺ + DA, compartment-specific)
        j_ac_coinc = da_mask * V_coinc * f_ca * shock_on

        # Total AC
        j_ac = bouton_mask * V_basal + j_ac_ca + j_ac_coinc
        dc = dc + j_ac

        # ---- PDE degradation (boutons only; 0 for dunce) ----
        j_pde = pde_mask * V_max_PDE * c_pos / (K_m + c_pos + 1e-30)
        dc = dc - j_pde

        # ---- cAMP Buffering (PKA-R subunits) ----
        B_free = jnp.maximum(B_total - cb_pos, 0.0)
        j_bind = k_on * c_pos * B_free
        j_unbind = k_off * cb_pos
        dc  = dc - j_bind + j_unbind
        dcb = j_bind - j_unbind

        # ---- Calcium with Rapid Buffering Approximation ----
        # Total Ca²⁺ influx has two components:
        #   1. Odour-evoked: sustained during odour presentation
        #      (KC depolarisation → voltage-gated Ca²⁺ channels)
        #   2. Shock-evoked: brief transient during each shock
        #      (circuit reverberation, antidromic activation)
        # RBA: only 1/(1+κ_S) ≈ 1.3% reaches free pool
        j_ca_total = stim_mask * (Ca_odor * odor_on + Ca_shock * shock_on)
        j_ca_free = j_ca_total * rba_factor

        # Decay back to resting [Ca²⁺] with τ = 46 ms
        dca = -(ca - Ca_rest) / tau_Ca + j_ca_free

        return jnp.concatenate([dc, dcb, dca])

    return diffrax.ODETerm(vector_field)


# =============================================================================
# 5. SENSOR FILTER (G-Flamp1 kinetics)
# =============================================================================

def sensor_lowpass(signal, dt, tau_sensor):
    """Causal exponential filter applied per-compartment.

    G-Flamp1 in vitro: t_on ≈ 0.20 s, t_off ≈ 0.087 s (Wang et al. 2022)
    In vivo response slower; bounds [0.5, 10] s.
    """
    a = 1.0 - jnp.exp(-dt / tau_sensor)

    def step_fn(F_prev, sig_now):
        F_new = F_prev + a * (sig_now - F_prev)
        return F_new, F_new

    F0 = signal[0]
    _, filtered = lax.scan(step_fn, F0, signal[1:])

    return jnp.concatenate([signal[:1], filtered], axis=0)


# =============================================================================
# 6. FORWARD MODEL
# =============================================================================

def solve_forward(params, geom, pairing_onsets, is_dunce,
                  t_save, solver_dt0=0.01, max_steps=200000):
    """Run model → apply sensor filter → return predicted ΔF/F at boutons.

    Note: solver_dt0 reduced to 0.01 (was 0.05) because τ_Ca = 46 ms
    requires finer initial steps for the implicit solver to track the
    fast Ca²⁺ transients accurately.
    """
    N = geom['n_total']
    Ca_rest = params[11]
    alpha = params[15]
    offset = params[16]
    tau_sensor = params[17]

    # Initial conditions
    y0 = jnp.zeros(3 * N)
    y0 = y0.at[2*N:3*N].set(Ca_rest)

    odor_fn, shock_fn = make_stim_signals(pairing_onsets)
    term = make_ode_term(geom, params, odor_fn, shock_fn, is_dunce)

    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(
        rtol=1e-5, atol=1e-7, dtmin=1e-6, dtmax=0.5)
    saveat = diffrax.SaveAt(ts=t_save)

    sol = diffrax.diffeqsolve(
        term, solver, t0=t_save[0], t1=t_save[-1], dt0=solver_dt0,
        y0=y0, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps, throw=False)

    # Extract bouton free cAMP
    c_free = sol.ys[:, :N]
    bouton_c = c_free[:, geom['bouton_idx']]  # [n_t, 5]

    # Raw signal
    raw_signal = alpha * bouton_c + offset

    # Apply G-Flamp1 sensor filter
    dt_save = t_save[1] - t_save[0]
    pred_dff = sensor_lowpass(raw_signal, dt_save, tau_sensor)

    return pred_dff


# =============================================================================
# 7. COST FUNCTION
# =============================================================================

def cost_single(fit_values, geom, pairing_onsets, t_data_wt, t_data_kd,
                data_wt, data_kd, sem_wt, sem_kd, weights, t_save):
    """Cost for one parameter vector (both genotypes)."""
    params = pack_full_params(fit_values)
    total = 0.0
    n_pts = 0.0

    for is_dunce_val, t_data, data_mean, data_sem in [
        (0.0, t_data_wt, data_wt, sem_wt),
        (1.0, t_data_kd, data_kd, sem_kd),
    ]:
        pred_dff = solve_forward(params, geom, pairing_onsets,
                                 is_dunce_val, t_save)

        for k in range(5):
            pred_k = pred_dff[:, k]
            t_dk = t_data[k]
            d_dk = data_mean[k]
            s_dk = data_sem[k]

            pred_interp = jnp.interp(t_dk, t_save, pred_k)
            valid = jnp.isfinite(d_dk)
            resid = jnp.where(valid,
                              weights[k] * (pred_interp - d_dk)**2 / (s_dk**2 + 1e-8),
                              0.0)
            total = total + jnp.sum(resid)
            n_pts = n_pts + jnp.sum(valid.astype(jnp.float64))

    return total / jnp.maximum(n_pts, 1.0)


# =============================================================================
# 8. GPU DIFFERENTIAL EVOLUTION
# =============================================================================

def differential_evolution_jax(cost_fn, bounds, popsize=20, maxiter=200,
                               mutation=(0.5, 1.5), crossover=0.8,
                               tol=1e-5, seed=42, verbose=True):
    n_dim = bounds.shape[0]
    pop_n = popsize * n_dim
    F_min, F_max = mutation
    key = jrandom.PRNGKey(seed)
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    key, subkey = jrandom.split(key)
    population = lo + (hi - lo) * jrandom.uniform(subkey, (pop_n, n_dim))

    costs = cost_fn(population)
    best_idx = jnp.argmin(costs)
    best_x = population[best_idx]
    best_cost = costs[best_idx]
    history = []

    for gen in range(maxiter):
        t0 = time_module.time()
        key, k1, k2, k3, k4, k5 = jrandom.split(key, 6)

        r1 = jrandom.randint(k1, (pop_n,), 0, pop_n)
        r2 = jrandom.randint(k2, (pop_n,), 0, pop_n)
        r3 = jrandom.randint(k3, (pop_n,), 0, pop_n)
        F = F_min + (F_max - F_min) * jrandom.uniform(k4, (pop_n, 1))

        donors = population[r1] + F * (population[r2] - population[r3])
        cross_mask = jrandom.uniform(k5, (pop_n, n_dim)) < crossover
        j_rand = jrandom.randint(k5, (pop_n,), 0, n_dim)
        force_mask = jax.nn.one_hot(j_rand, n_dim, dtype=jnp.bool_)
        cross_mask = cross_mask | force_mask

        trials = jnp.where(cross_mask, donors, population)
        trials = jnp.clip(trials, lo, hi)
        trial_costs = cost_fn(trials)

        improved = trial_costs < costs
        population = jnp.where(improved[:, None], trials, population)
        costs = jnp.where(improved, trial_costs, costs)

        gen_best_idx = jnp.argmin(costs)
        gen_best_cost = costs[gen_best_idx]
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_x = population[gen_best_idx]

        elapsed = time_module.time() - t0
        history.append(float(best_cost))

        if verbose and (gen % 5 == 0 or gen == maxiter - 1):
            cost_std = float(jnp.std(costs))
            print(f"  Gen {gen:4d}/{maxiter}: best={float(best_cost):.6f}  "
                  f"mean={float(jnp.mean(costs)):.4f}  "
                  f"std={cost_std:.6f}  ({elapsed:.2f}s/gen)")
            if cost_std < tol and gen > 20:
                print(f"  Converged (std < {tol})")
                break

    return np.array(best_x), float(best_cost), history


# =============================================================================
# 9. DATA LOADING
# =============================================================================

def load_data(csv_path: str) -> Dict:
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


def prepare_data_arrays(data: Dict, dt_ds: float = 2.0,
                        t_fit_lo: float = -5.0, t_fit_hi: float = 80.0):
    """Prepare padded JAX arrays."""
    comps = ['g1', 'g2', 'g3', 'g4', 'g5']

    def process_geno(geno):
        arrays_t, arrays_m, arrays_s = [], [], []
        for comp in comps:
            key = (geno, comp)
            if key in data:
                t, m, s = data[key]
                mask = (t >= t_fit_lo) & (t <= t_fit_hi)
                t, m, s = t[mask], m[mask], s[mask]
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
# 10. PUBLICATION FIGURE
# =============================================================================

COMP_COLORS = {'g1': '#e41a1c', 'g2': '#ff7f00', 'g3': '#999999',
               'g4': '#377eb8', 'g5': '#4daf4a'}
COMP_MAP = {'g1': 0, 'g2': 1, 'g3': 2, 'g4': 3, 'g5': 4}

WT_COLOR  = '#4daf4a'
KD_COLOR  = '#ff7f00'
WT_FACE   = '#a6d96a'
KD_FACE   = '#fdae61'


def run_model_np(params_array, pairing_onsets, is_dunce, t_save_np, geom):
    """Run model returning numpy arrays."""
    pred = solve_forward(
        jnp.array(params_array), geom, jnp.array(pairing_onsets),
        float(is_dunce), jnp.array(t_save_np))
    return np.array(t_save_np), np.array(pred)


def plot_figure2_style(params_array, data_full, pairing_onsets, geom,
                       save_dir='.', tag='', t_display=(-15, 90)):
    """Publication figure: 5 panels (γ1→γ5), dnc-wt + dnc-KD + model."""
    t_lo, t_hi = t_display
    t_model = np.arange(-5.0, 82.0, 0.25)
    comps = ['g1', 'g2', 'g3', 'g4', 'g5']

    model_traces = {}
    for geno, is_d in [('dnc-wt', 0.0), ('dnc-KD', 1.0)]:
        t_m, pred = run_model_np(params_array, pairing_onsets, is_d, t_model, geom)
        model_traces[geno] = (t_m, pred)

    fig, axes = plt.subplots(5, 1, figsize=(7, 16), sharex=True)

    for row, comp in enumerate(comps):
        ax = axes[row]
        ci = COMP_MAP[comp]

        for geno, color, face_color, label in [
            ('dnc-wt', WT_COLOR, WT_FACE, 'dnc-wt'),
            ('dnc-KD', KD_COLOR, KD_FACE, 'dnc-KD'),
        ]:
            key = (geno, comp)
            if key in data_full:
                t_d, m_d, s_d = data_full[key]
                mask = (t_d >= t_lo) & (t_d <= t_hi)
                ax.fill_between(t_d[mask], m_d[mask] - s_d[mask],
                                m_d[mask] + s_d[mask],
                                alpha=0.25, color=face_color, linewidth=0)
                ax.plot(t_d[mask], m_d[mask], '-', color=color,
                       linewidth=1.5, label=label)

            t_m, pred = model_traces[geno]
            m_mask = (t_m >= t_lo) & (t_m <= t_hi)
            ax.plot(t_m[m_mask], pred[m_mask, ci], '--', color=color,
                   linewidth=2.0, alpha=0.8)

        for onset in pairing_onsets:
            if t_lo <= onset <= t_hi:
                ax.plot(onset, -0.05, marker='^', markersize=4,
                       color='black', clip_on=False)

        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.set_ylabel('ΔF/F₀', fontsize=11)
        ax.set_ylim(-0.3, 1.5)
        ax.text(0.02, 0.92, f'γ{row+1}', transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')

        if row == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    axes[-1].set_xlim(t_lo, t_hi)

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.3)

    path = os.path.join(save_dir, f'figure2_style{tag}.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 2-style plot: {path}")

    path_pdf = os.path.join(save_dir, f'figure2_style{tag}.pdf')
    fig2, axes2 = plt.subplots(5, 1, figsize=(7, 16), sharex=True)
    for row, comp in enumerate(comps):
        ax = axes2[row]
        ci = COMP_MAP[comp]
        for geno, color, face_color, label in [
            ('dnc-wt', WT_COLOR, WT_FACE, 'dnc-wt'),
            ('dnc-KD', KD_COLOR, KD_FACE, 'dnc-KD'),
        ]:
            key = (geno, comp)
            if key in data_full:
                t_d, m_d, s_d = data_full[key]
                mask = (t_d >= t_lo) & (t_d <= t_hi)
                ax.fill_between(t_d[mask], m_d[mask]-s_d[mask],
                                m_d[mask]+s_d[mask], alpha=0.25,
                                color=face_color, linewidth=0)
                ax.plot(t_d[mask], m_d[mask], '-', color=color,
                       linewidth=1.5, label=label)
            t_m, pred = model_traces[geno]
            m_mask = (t_m >= t_lo) & (t_m <= t_hi)
            ax.plot(t_m[m_mask], pred[m_mask, ci], '--', color=color,
                   linewidth=2.0, alpha=0.8)
        for onset in pairing_onsets:
            if t_lo <= onset <= t_hi:
                ax.plot(onset, -0.05, marker='^', markersize=4,
                       color='black', clip_on=False)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.set_ylabel('ΔF/F₀', fontsize=11)
        ax.set_ylim(-0.3, 1.5)
        ax.text(0.02, 0.92, f'γ{row+1}', transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
        if row == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
    axes2[-1].set_xlabel('Time (s)', fontsize=12)
    axes2[-1].set_xlim(t_lo, t_hi)
    fig2.align_ylabels(axes2)
    fig2.tight_layout(h_pad=0.3)
    fig2.savefig(path_pdf, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved PDF: {path_pdf}")


def plot_diagnostic(params_array, data_full, pairing_onsets, geom,
                    save_dir='.', tag=''):
    """Multi-panel diagnostic."""
    t_model = np.arange(-5.0, 82.0, 0.5)
    comps = ['g1', 'g2', 'g3', 'g4', 'g5']

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    for col, (geno, is_d, title) in enumerate([
        ('dnc-wt', 0.0, 'Wild-Type'),
        ('dnc-KD', 1.0, 'dunce-KD'),
    ]):
        t_m, pred = run_model_np(params_array, pairing_onsets, is_d, t_model, geom)

        ax = fig.add_subplot(gs[0, col])
        for comp in comps:
            ci = COMP_MAP[comp]
            color = COMP_COLORS[comp]
            key = (geno, comp)
            if key in data_full:
                td, md, sd = data_full[key]
                mask = (td >= -5) & (td <= 80)
                ax.fill_between(td[mask], md[mask]-sd[mask], md[mask]+sd[mask],
                                alpha=0.15, color=color)
                ax.plot(td[mask], md[mask], '-', color=color, alpha=0.5, linewidth=0.8)
            ax.plot(t_m, pred[:, ci], '--', color=color, linewidth=2, label=f'{comp}')
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('ΔF/F₀')
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        ax.set_xlim(-5, 80)
        for o in pairing_onsets:
            ax.axvline(o, color='red', alpha=0.1, linewidth=0.5)

    ax = fig.add_subplot(gs[1, :])
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
            mask = (td >= -5) & (td <= 80)
            ci = COMP_MAP[comp]
            is_d = 1.0 if geno == 'dnc-KD' else 0.0
            t_m, pred = run_model_np(params_array, pairing_onsets, is_d, t_model, geom)
            pred_i = np.interp(td[mask], t_m, pred[:, ci])
            rmses.append(np.sqrt(np.mean((pred_i - md[mask])**2)))
        ax.bar(x_pos + gi*width, rmses, width, label=geno, alpha=0.7,
               color=[WT_COLOR, KD_COLOR][gi])
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels([f'γ{i+1}' for i in range(5)])
    ax.set_ylabel('RMSE (ΔF/F)')
    ax.set_title('Fit Quality per Compartment')
    ax.legend()

    fig.suptitle('cAMP Model v5 — RBA Ca²⁺ + Free K_Ca + Two-Term AC + DA Mask', fontsize=13, fontweight='bold')
    path = os.path.join(save_dir, f'fit_diagnostic{tag}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved diagnostic: {path}")


def plot_convergence(history, save_dir='.'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, 'k-', linewidth=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Cost')
    ax.set_title('Optimisation Convergence')
    ax.set_yscale('log')
    fig.tight_layout()
    path = os.path.join(save_dir, 'convergence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved convergence: {path}")


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPU cAMP model optimisation v5 (RBA Ca²⁺ + free K_Ca)')
    parser.add_argument('--data', type=str, default='all_camp_long.csv')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--popsize', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--plot-only', type=str, default=None, metavar='JSON')
    parser.add_argument('--stim-onsets', type=str, default=None)
    parser.add_argument('--downsample-dt', type=float, default=2.0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--t-display', type=str, default='-15,90',
                        help='Time window for Figure 2 plot (lo,hi)')
    args = parser.parse_args()

    if args.cpu:
        jax.config.update('jax_platform_name', 'cpu')
        print("Forced CPU mode")

    os.makedirs(args.outdir, exist_ok=True)

    if args.stim_onsets:
        pairing_onsets = np.array([float(x) for x in args.stim_onsets.split(',')])
    else:
        pairing_onsets = np.arange(0.0, 60.0, 5.0)
    print(f"Pairing onsets: {pairing_onsets.tolist()}")

    t_display = tuple(float(x) for x in args.t_display.split(','))

    geom = build_geometry()
    print(f"Grid: {geom['n_total']} bins")
    print(f"DA mask at boutons: {[float(geom['da_mask'][bi]) for bi in geom['bouton_idx']]}")

    # Print Ca²⁺ RBA info
    kappa_S = float(DEFAULTS[14])
    tau_Ca = float(DEFAULTS[13])
    print(f"Ca²⁺ RBA: κ_S = {kappa_S:.0f}, τ_Ca = {tau_Ca*1000:.1f} ms, "
          f"free fraction = {1/(1+kappa_S):.4f} ({1/(1+kappa_S)*100:.1f}%)")

    print(f"Loading: {args.data}")
    data_full = load_data(args.data)
    print(f"  {len(data_full)} traces")

    weights = jnp.array([1.0, 1.5, 1.0, 2.0, 2.0])

    if args.plot_only:
        with open(args.plot_only) as f:
            saved = json.load(f)
        params_array = np.array([saved['all_params'][n] for n in PARAM_NAMES])
        if 'stim_onsets' in saved:
            pairing_onsets = np.array(saved['stim_onsets'])
        plot_figure2_style(params_array, data_full, pairing_onsets, geom,
                           save_dir=args.outdir, t_display=t_display)
        plot_diagnostic(params_array, data_full, pairing_onsets, geom,
                        save_dir=args.outdir)
        return

    # Prepare data
    t_wt, t_kd, m_wt, m_kd, s_wt, s_kd = prepare_data_arrays(
        data_full, dt_ds=args.downsample_dt)
    t_save = jnp.arange(-5.0, 82.0, 0.5)   # finer grid for 46ms transients
    pairing_j = jnp.array(pairing_onsets)

    @jit
    def evaluate_population(population):
        return vmap(
            lambda x: cost_single(x, geom, pairing_j, t_wt, t_kd,
                                   m_wt, m_kd, s_wt, s_kd, weights, t_save)
        )(population)

    print("\nJIT compiling (may take 1-3 min with fast Ca²⁺ dynamics)...")
    t0 = time_module.time()
    dummy = jnp.tile(DEFAULTS[FIT_INDICES], (3, 1))
    _ = evaluate_population(dummy).block_until_ready()
    print(f"JIT done in {time_module.time()-t0:.1f}s")

    t0 = time_module.time()
    bench = jnp.tile(DEFAULTS[FIT_INDICES], (20, 1))
    _ = evaluate_population(bench).block_until_ready()
    per_eval = (time_module.time()-t0) / 20
    print(f"Benchmark: {per_eval*1000:.1f} ms / forward pass")

    maxiter = 50 if args.quick else args.maxiter

    print(f"\n{'='*60}")
    print(f"GPU DIFFERENTIAL EVOLUTION v5 — RBA Ca²⁺ + Free K_Ca")
    print(f"{'='*60}")
    print(f"Parameters ({len(FIT_NAMES)}): {FIT_NAMES}")
    print(f"  ↳ V_Ca:      Ca²⁺-only rutabaga (all compartments during odour)")
    print(f"  ↳ V_coinc:   Coincidence Ca²⁺+DA (γ1-γ2 full, γ3 partial)")
    print(f"  ↳ K_Ca:      [{float(FIT_BOUNDS[5,0])}, {float(FIT_BOUNDS[5,1])}] μM  ← NOW FREE")
    print(f"  ↳ Ca_odor:   Total Ca²⁺ influx during odour (RBA scales to free)")
    print(f"  ↳ Ca_shock:  Total Ca²⁺ influx during shock (brief transient)")
    print(f"  ↳ DA mask:   {[float(geom['da_mask'][bi]) for bi in geom['bouton_idx']]}")
    print(f"  ↳ κ_S = {kappa_S:.0f} (fixed), τ_Ca = {tau_Ca*1000:.0f} ms (fixed)")
    print(f"  ↳ τ_sensor:  [{float(FIT_BOUNDS[-1,0])}, {float(FIT_BOUNDS[-1,1])}] s")
    print(f"Pop: {args.popsize}×{len(FIT_NAMES)} = {args.popsize*len(FIT_NAMES)}")
    print(f"Generations: {maxiter}")
    print(f"Device: {jax.devices()[0]}")
    print(f"{'='*60}\n")

    t_total = time_module.time()

    best_x, best_cost, history = differential_evolution_jax(
        evaluate_population,
        bounds=np.array(FIT_BOUNDS),
        popsize=args.popsize,
        maxiter=maxiter,
        seed=args.seed,
        verbose=True,
    )

    total_time = time_module.time() - t_total

    full_params = np.array(DEFAULTS)
    for i, fi in enumerate(FIT_INDICES):
        full_params[int(fi)] = best_x[i]

    print(f"\n{'='*60}")
    print(f"COMPLETE ({total_time:.0f}s = {total_time/60:.1f} min)")
    print(f"{'='*60}")
    print(f"Best cost: {best_cost:.6f}\n")

    print("Fitted parameters:")
    at_bound_count = 0
    for name, val in zip(FIT_NAMES, best_x):
        idx_in_fit = FIT_NAMES.index(name)
        lo, hi = float(FIT_BOUNDS[idx_in_fit, 0]), float(FIT_BOUNDS[idx_in_fit, 1])
        at_lo = abs(val - lo) < 0.01 * (hi - lo)
        at_hi = abs(val - hi) < 0.01 * (hi - lo)
        flag = ' ← AT BOUND' if (at_lo or at_hi) else ''
        if at_lo or at_hi:
            at_bound_count += 1
        print(f"  {name:20s} = {val:10.4f}  [{lo:.3f}, {hi:.3f}]{flag}")

    print(f"\n  Parameters at bounds: {at_bound_count}/{len(FIT_NAMES)}")

    # Fixed parameters used
    print(f"\nFixed Ca²⁺ parameters (Bhatt & Bhatt 2011, Drosophila Ib terminal):")
    print(f"  κ_S:           {full_params[14]:.0f}")
    print(f"  τ_Ca:          {full_params[13]*1000:.1f} ms")
    print(f"  Free fraction: {1/(1+full_params[14])*100:.1f}%")

    # Derived quantities
    D_eff = full_params[0] / (1 + full_params[1] / full_params[2])
    eta = full_params[8] / (4 * np.pi * 2.5e-3 * D_eff * full_params[9])
    coinc_ratio = full_params[18] / full_params[5] if full_params[5] > 0 else float('inf')
    ca_free_odor = full_params[12] / (1 + full_params[14])
    ca_free_shock = full_params[19] / (1 + full_params[14])
    K_Ca_fit = full_params[6]
    # Estimate peak free [Ca²⁺] during shock
    ca_peak = full_params[11] + ca_free_shock * full_params[13]
    f_ca_peak = ca_peak**2 / (K_Ca_fit**2 + ca_peak**2)
    print(f"\nDerived:")
    print(f"  D_eff:           {D_eff:.1f} μm²/s")
    print(f"  V_max_PDE:       {full_params[10]*full_params[8]:.2f} μM/s")
    print(f"  η:               {eta:.3f}")
    print(f"  τ_sensor:        {full_params[17]:.2f} s")
    print(f"  V_coinc/V_Ca:    {coinc_ratio:.1f}×  (supralinearity ratio)")
    print(f"  Ca_free (odour): {ca_free_odor:.4f} μM/s  (after RBA)")
    print(f"  Ca_free (shock): {ca_free_shock:.4f} μM/s  (after RBA)")
    print(f"  K_Ca (fitted):   {K_Ca_fit*1000:.1f} nM")
    print(f"  Peak free [Ca²⁺] during shock: ~{ca_peak*1000:.0f} nM")
    print(f"  f(Ca) at peak:   {f_ca_peak:.3f}  (Hill activation fraction)")

    # Save
    result = {
        'cost': best_cost,
        'param_names': FIT_NAMES,
        'param_values': best_x.tolist(),
        'all_params': {n: float(full_params[i]) for i, n in enumerate(PARAM_NAMES)},
        'stim_onsets': pairing_onsets.tolist(),
        'da_mask': [1.0, 1.0, 0.5, 0.0, 0.0],
        'fixed_params': {
            'kappa_S': float(full_params[14]),
            'tau_Ca_ms': float(full_params[13] * 1000),
            'source': 'Bhatt & Bhatt 2011, Drosophila Ib motor terminal',
        },
        'history': history,
        'total_time_s': total_time,
        'device': str(jax.devices()[0]),
        'model_version': 'v5_RBA_free_KCa',
    }
    json_path = os.path.join(args.outdir, 'best_params.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {json_path}")

    plot_figure2_style(full_params, data_full, pairing_onsets, geom,
                       save_dir=args.outdir, t_display=t_display)
    plot_diagnostic(full_params, data_full, pairing_onsets, geom,
                    save_dir=args.outdir)
    plot_convergence(history, save_dir=args.outdir)

    print(f"\nAll outputs in: {args.outdir}/")


if __name__ == '__main__':
    main()
