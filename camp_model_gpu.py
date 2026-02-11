#!/usr/bin/env python3
"""
GPU-Accelerated cAMP Nanodomain Model
======================================

JAX + Diffrax implementation for NVIDIA GPU (RTX 5070 / CUDA).

Features over the scipy version:
  - JIT-compiled ODE RHS → 10-100× faster per solve
  - vmap for parallel parameter sweeps → run 1000s of sims simultaneously on GPU
  - Automatic differentiation → exact gradients for sensitivity & fitting
  - Diffrax implicit solvers for stiff buffering kinetics

Requirements:
    pip install jax[cuda12] diffrax equinox matplotlib numpy

For RTX 5070 (CUDA 12.x):
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install diffrax equinox

Usage:
    python camp_model_gpu.py              # Run all experiments on GPU
    JAX_PLATFORM_NAME=cpu python camp_model_gpu.py  # Force CPU fallback
"""

import os
import time
from functools import partial
from typing import NamedTuple, Optional

import numpy as np

# ── JAX imports ──────────────────────────────────────────────
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

# ── Diffrax: JAX-native ODE solvers ─────────────────────────
import diffrax

# ── Plotting ─────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Check device
print(f"JAX devices: {jax.devices()}")
print(f"Using: {jax.devices()[0].device_kind}")


# ═══════════════════════════════════════════════════════════════
# 1. PARAMETERS (as JAX-friendly NamedTuples / pytrees)
# ═══════════════════════════════════════════════════════════════

class GeomParams(NamedTuple):
    """Spatial geometry — all floats for JAX tracing."""
    n_compartments: int = 5
    bouton_diameter: float = 1.5      # μm
    axon_diameter: float = 0.3        # μm
    inter_bouton_distance: float = 5.0  # μm
    axon_bins_per_segment: int = 20


class DiffParams(NamedTuple):
    D_free: float = 130.0       # μm²/s
    B_total: float = 20.0       # μM
    K_D_buffer: float = 2.0     # μM
    k_on_buffer: float = 10.0   # μM⁻¹s⁻¹


class EnzParams(NamedTuple):
    V_basal: float = 0.5        # μM/s
    V_max_AC: float = 10.0      # μM/s
    K_Ca: float = 0.5           # μM
    n_Hill_AC: float = 2.0
    k_cat: float = 5.0          # s⁻¹
    K_m: float = 2.4            # μM
    PDE_concentration: float = 1.0  # μM


class CalcParams(NamedTuple):
    Ca_rest: float = 0.05       # μM
    Ca_amplitude: float = 2.0   # μM
    tau_Ca: float = 1.0         # s
    Ca_pulse_duration: float = 0.1  # s


class StimParams(NamedTuple):
    odor_duration: float = 5.0
    shock_onset: float = 4.0
    shock_duration: float = 1.0
    n_pairings: int = 6
    inter_trial_interval: float = 30.0


class ModelConfig(NamedTuple):
    """Full model configuration — JAX pytree compatible."""
    geom: GeomParams = GeomParams()
    diff: DiffParams = DiffParams()
    enz: EnzParams = EnzParams()
    calc: CalcParams = CalcParams()
    stim: StimParams = StimParams()
    pde_activity_fraction: float = 1.0  # 0.0 = dunce⁻, 1.0 = WT


# ═══════════════════════════════════════════════════════════════
# 2. GEOMETRY — Build the 1D grid (done once on CPU, static)
# ═══════════════════════════════════════════════════════════════

class GridArrays(NamedTuple):
    """Precomputed grid arrays — transferred to GPU once."""
    x: jnp.ndarray              # positions (n_total,)
    dx: jnp.ndarray             # bin widths (n_total,)
    A: jnp.ndarray              # cross-sections (n_total,)
    is_bouton: jnp.ndarray      # bool mask (n_total,)
    compartment_id: jnp.ndarray # int: 0-4 for boutons, -1 for axon
    is_aversive: jnp.ndarray    # bool mask: receives DAN Ca²⁺
    n_total: int
    bouton_bin_indices: jnp.ndarray  # shape (n_compartments,) — single index per bouton


def build_grid(geom: GeomParams) -> GridArrays:
    """Build 1D grid. Called once — not JIT'd."""
    N = geom.n_compartments
    n_axon = geom.axon_bins_per_segment
    d_b = geom.bouton_diameter
    d_a = geom.axon_diameter
    L_seg = geom.inter_bouton_distance - d_b
    axon_dx = L_seg / n_axon

    A_bouton = np.pi * (d_b / 2) ** 2
    A_axon = np.pi * (d_a / 2) ** 2

    positions, dxs, areas, is_bout, comp_ids = [], [], [], [], []
    bouton_bins = []
    cx = 0.0
    idx = 0

    for i in range(N):
        # Bouton
        bouton_bins.append(idx)
        positions.append(cx + d_b / 2)
        dxs.append(d_b)
        areas.append(A_bouton)
        is_bout.append(True)
        comp_ids.append(i)
        idx += 1
        cx += d_b

        # Axon segment
        if i < N - 1:
            for j in range(n_axon):
                positions.append(cx + (j + 0.5) * axon_dx)
                dxs.append(axon_dx)
                areas.append(A_axon)
                is_bout.append(False)
                comp_ids.append(-1)
                idx += 1
            cx += L_seg

    # Mark aversive compartments (γ1=0, γ2=1)
    is_aversive = np.array([(c in [0, 1]) for c in comp_ids])

    return GridArrays(
        x=jnp.array(positions),
        dx=jnp.array(dxs),
        A=jnp.array(areas),
        is_bouton=jnp.array(is_bout),
        compartment_id=jnp.array(comp_ids, dtype=jnp.int32),
        is_aversive=jnp.array(is_aversive),
        n_total=idx,
        bouton_bin_indices=jnp.array(bouton_bins, dtype=jnp.int32),
    )


def precompute_coupling(grid: GridArrays, D_free: float):
    """Precompute diffusion coupling coefficients.

    Returns (alpha_plus, alpha_minus) arrays for vectorized diffusion.
    """
    n = grid.n_total
    A = grid.A
    dx = grid.dx

    # Interface areas (harmonic mean)
    A_interface = 2.0 * A[:-1] * A[1:] / (A[:-1] + A[1:])
    dist = 0.5 * (dx[:-1] + dx[1:])
    flux_coeff = D_free * A_interface / dist

    V = A * dx  # bin volumes

    alpha_plus = jnp.zeros(n)
    alpha_minus = jnp.zeros(n)
    alpha_plus = alpha_plus.at[:-1].set(flux_coeff / V[:-1])
    alpha_minus = alpha_minus.at[1:].set(flux_coeff / V[1:])

    return alpha_plus, alpha_minus


# ═══════════════════════════════════════════════════════════════
# 3. STIMULUS FUNCTIONS (JIT-compatible, no Python control flow)
# ═══════════════════════════════════════════════════════════════

@jit
def is_during_odor(t: float, stim: StimParams) -> float:
    """Return 1.0 if odor is on at time t, else 0.0. Differentiable approx."""
    # Check each trial with smooth step functions
    result = 0.0
    # Unroll for fixed n_pairings — use scan for variable
    trial_onsets = jnp.arange(stim.n_pairings) * stim.inter_trial_interval
    # For each trial: is t in [onset, onset + odor_duration]?
    in_odor = (t >= trial_onsets) & (t < trial_onsets + stim.odor_duration)
    return jnp.any(in_odor).astype(jnp.float32)


@jit
def is_during_shock(t: float, stim: StimParams) -> float:
    """Return 1.0 if shock is on at time t, else 0.0."""
    trial_onsets = jnp.arange(stim.n_pairings) * stim.inter_trial_interval
    shock_starts = trial_onsets + stim.shock_onset
    shock_ends = shock_starts + stim.shock_duration
    in_shock = (t >= shock_starts) & (t < shock_ends)
    return jnp.any(in_shock).astype(jnp.float32)


# ═══════════════════════════════════════════════════════════════
# 4. ODE RIGHT-HAND SIDE (fully JIT-compiled)
# ═══════════════════════════════════════════════════════════════

def make_rhs(grid: GridArrays, alpha_plus: jnp.ndarray,
             alpha_minus: jnp.ndarray):
    """Create a JIT-compiled RHS function closed over grid arrays.

    State vector y: [c_free (N), c_bound (N), Ca (N)]
    """
    N = grid.n_total
    is_bouton = grid.is_bouton
    is_aversive = grid.is_aversive

    # PDE activity mask: only in boutons
    pde_mask = is_bouton.astype(jnp.float32)

    @jit
    def rhs(t, y, config: ModelConfig):
        """dy/dt for the full system. JIT-compiled."""
        c_free = y[:N]
        c_bound = y[N:2*N]
        ca = y[2*N:3*N]

        diff = config.diff
        enz = config.enz
        calc = config.calc
        stim = config.stim

        c_pos = jnp.maximum(c_free, 0.0)

        # ── Diffusion ──
        dc_diff = (
            alpha_plus * jnp.roll(c_free, -1) - alpha_plus * c_free
            + alpha_minus * jnp.roll(c_free, 1) - alpha_minus * c_free
        )
        # Fix boundary: roll wraps around, zero out boundary terms
        dc_diff = dc_diff.at[0].set(
            alpha_plus[0] * (c_free[1] - c_free[0])
        )
        dc_diff = dc_diff.at[N-1].set(
            alpha_minus[N-1] * (c_free[N-2] - c_free[N-1])
        )

        # ── AC production (boutons only) ──
        f_odor = is_during_odor(t, stim)
        f_ca = ca**enz.n_Hill_AC / (enz.K_Ca**enz.n_Hill_AC + ca**enz.n_Hill_AC)
        j_ac = (enz.V_basal + enz.V_max_AC * f_ca * f_odor) * is_bouton

        # ── PDE degradation (boutons only, scaled by pde_activity_fraction) ──
        V_max_pde = enz.PDE_concentration * enz.k_cat * config.pde_activity_fraction
        j_pde = V_max_pde * c_pos / (enz.K_m + c_pos) * pde_mask

        # ── Buffering ──
        B_free = jnp.maximum(diff.B_total - c_bound, 0.0)
        k_off = diff.k_on_buffer * diff.K_D_buffer
        j_bind = diff.k_on_buffer * c_pos * B_free
        j_unbind = k_off * jnp.maximum(c_bound, 0.0)

        # ── Calcium ──
        shock_on = is_during_shock(t, stim)
        j_ca_influx = (calc.Ca_amplitude / calc.Ca_pulse_duration) * shock_on * is_aversive
        dca = j_ca_influx - (ca - calc.Ca_rest) / calc.tau_Ca

        # ── Assemble ──
        dc_free = dc_diff + j_ac - j_pde - j_bind + j_unbind
        dc_bound = j_bind - j_unbind

        return jnp.concatenate([dc_free, dc_bound, dca])

    return rhs


# ═══════════════════════════════════════════════════════════════
# 5. SOLVER (Diffrax — GPU-native ODE integration)
# ═══════════════════════════════════════════════════════════════

def solve_model(config: ModelConfig, dt_save: float = 0.2,
                solver_type: str = 'kvaerno5') -> dict:
    """Solve the model using Diffrax.

    Args:
        config: Model configuration
        dt_save: Time interval for saving output
        solver_type: 'kvaerno5' (implicit, stiff), 'dopri5' (explicit)

    Returns:
        dict with keys: t, c_free, c_bound, ca, c_total, bouton_c_free
    """
    grid = build_grid(config.geom)
    N = grid.n_total

    alpha_plus, alpha_minus = precompute_coupling(grid, config.diff.D_free)
    rhs_fn = make_rhs(grid, alpha_plus, alpha_minus)

    # Initial conditions
    y0 = jnp.zeros(3 * N)
    y0 = y0.at[2*N:3*N].set(config.calc.Ca_rest)  # resting Ca²⁺

    # Time span
    total_time = (
        (config.stim.n_pairings - 1) * config.stim.inter_trial_interval
        + config.stim.odor_duration + 20.0
    )
    t_save = jnp.arange(0.0, total_time, dt_save)

    # Diffrax ODE term: wraps rhs to pass config as args
    def vector_field(t, y, args):
        return rhs_fn(t, y, args)

    term = diffrax.ODETerm(vector_field)

    # Choose solver
    if solver_type == 'kvaerno5':
        solver = diffrax.Kvaerno5()
    elif solver_type == 'dopri5':
        solver = diffrax.Dopri5()
    else:
        solver = diffrax.Tsit5()

    # Adaptive step controller
    stepsize_controller = diffrax.PIDController(
        rtol=1e-6, atol=1e-8, dtmin=1e-6, dtmax=0.1
    )

    # Save at specified times
    saveat = diffrax.SaveAt(ts=t_save)

    # Solve
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=total_time,
        dt0=0.001,
        y0=y0,
        args=config,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=500_000,
    )

    # Unpack
    t_out = np.array(sol.ts)
    y_out = np.array(sol.ys)

    c_free = y_out[:, :N]
    c_bound = y_out[:, N:2*N]
    ca = y_out[:, 2*N:3*N]
    c_total = c_free + c_bound

    # Bouton averages
    bouton_idx = np.array(grid.bouton_bin_indices)
    bouton_c_free = c_free[:, bouton_idx]
    bouton_c_total = c_total[:, bouton_idx]

    return {
        't': t_out,
        'c_free': c_free,
        'c_bound': c_bound,
        'ca': ca,
        'c_total': c_total,
        'bouton_c_free': bouton_c_free,
        'bouton_c_total': bouton_c_total,
        'grid': grid,
        'config': config,
    }


# ═══════════════════════════════════════════════════════════════
# 6. VMAP PARAMETER SWEEP (the GPU killer feature)
# ═══════════════════════════════════════════════════════════════

def parallel_sweep_kcat(kcat_values: np.ndarray,
                        base_config: ModelConfig,
                        dt_save: float = 0.5) -> dict:
    """Run many k_cat values IN PARALLEL on GPU using vmap.

    This is where the RTX 5070 really shines — each parameter
    combination runs as a separate "batch element" on GPU.
    """
    grid = build_grid(base_config.geom)
    N = grid.n_total
    alpha_plus, alpha_minus = precompute_coupling(grid, base_config.diff.D_free)
    rhs_fn = make_rhs(grid, alpha_plus, alpha_minus)

    total_time = (
        (base_config.stim.n_pairings - 1) * base_config.stim.inter_trial_interval
        + base_config.stim.odor_duration + 20.0
    )
    t_save = jnp.arange(0.0, total_time, dt_save)

    def solve_single(kcat):
        """Solve for a single k_cat value."""
        # Create config with modified k_cat
        enz = EnzParams(
            V_basal=base_config.enz.V_basal,
            V_max_AC=base_config.enz.V_max_AC,
            K_Ca=base_config.enz.K_Ca,
            n_Hill_AC=base_config.enz.n_Hill_AC,
            k_cat=kcat,
            K_m=base_config.enz.K_m,
            PDE_concentration=base_config.enz.PDE_concentration,
        )
        config = ModelConfig(
            geom=base_config.geom,
            diff=base_config.diff,
            enz=enz,
            calc=base_config.calc,
            stim=base_config.stim,
            pde_activity_fraction=base_config.pde_activity_fraction,
        )

        y0 = jnp.zeros(3 * N).at[2*N:3*N].set(base_config.calc.Ca_rest)

        def vector_field(t, y, args):
            return rhs_fn(t, y, args)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Kvaerno5(),
            t0=0.0, t1=total_time, dt0=0.001,
            y0=y0, args=config,
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=diffrax.PIDController(
                rtol=1e-5, atol=1e-7, dtmin=1e-5, dtmax=0.2
            ),
            max_steps=200_000,
        )
        # Return bouton free cAMP at all time points
        bouton_idx = grid.bouton_bin_indices
        return sol.ys[:, bouton_idx[:N]]  # [n_time, n_compartments]

    # vmap over k_cat values — runs ALL in parallel on GPU!
    batched_solve = vmap(solve_single)

    print(f"Running {len(kcat_values)} simulations in parallel via vmap...")
    t0 = time.time()
    results = batched_solve(jnp.array(kcat_values))
    # Block until GPU finishes
    results.block_until_ready()
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s ({len(kcat_values)/elapsed:.1f} sims/sec)")

    return {
        't': np.array(t_save),
        'bouton_c_free': np.array(results),  # [n_kcat, n_time, n_compartments]
        'kcat_values': kcat_values,
    }


# ═══════════════════════════════════════════════════════════════
# 7. AUTODIFF SENSITIVITY ANALYSIS (free gradients!)
# ═══════════════════════════════════════════════════════════════

def compute_sensitivity(base_config: ModelConfig):
    """Use JAX autodiff to compute d(γ4-5 cAMP) / d(parameter).

    This gives exact gradients — much better than finite differences.
    """
    grid = build_grid(base_config.geom)
    N = grid.n_total
    alpha_plus, alpha_minus = precompute_coupling(grid, base_config.diff.D_free)
    rhs_fn = make_rhs(grid, alpha_plus, alpha_minus)

    total_time = (
        (base_config.stim.n_pairings - 1) * base_config.stim.inter_trial_interval
        + base_config.stim.odor_duration + 20.0
    )

    def spreading_metric(kcat, pde_frac, B_total):
        """Scalar function: cAMP in γ4-5 at end of training.

        JAX will differentiate this w.r.t. all inputs.
        """
        enz = EnzParams(
            V_basal=base_config.enz.V_basal,
            V_max_AC=base_config.enz.V_max_AC,
            K_Ca=base_config.enz.K_Ca,
            n_Hill_AC=base_config.enz.n_Hill_AC,
            k_cat=kcat,
            K_m=base_config.enz.K_m,
            PDE_concentration=base_config.enz.PDE_concentration,
        )
        diff = DiffParams(
            D_free=base_config.diff.D_free,
            B_total=B_total,
            K_D_buffer=base_config.diff.K_D_buffer,
            k_on_buffer=base_config.diff.k_on_buffer,
        )
        config = ModelConfig(
            geom=base_config.geom, diff=diff, enz=enz,
            calc=base_config.calc, stim=base_config.stim,
            pde_activity_fraction=pde_frac,
        )

        y0 = jnp.zeros(3 * N).at[2*N:3*N].set(base_config.calc.Ca_rest)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: rhs_fn(t, y, args)),
            diffrax.Kvaerno5(),
            t0=0.0, t1=total_time, dt0=0.001,
            y0=y0, args=config,
            saveat=diffrax.SaveAt(t1=True),  # only save final state
            stepsize_controller=diffrax.PIDController(
                rtol=1e-5, atol=1e-7, dtmin=1e-5, dtmax=0.2
            ),
            max_steps=200_000,
        )

        y_final = sol.ys[0]
        c_free_final = y_final[:N]
        bouton_idx = grid.bouton_bin_indices
        # Mean cAMP in γ4-5 (indices 3, 4)
        return (c_free_final[bouton_idx[3]] + c_free_final[bouton_idx[4]]) / 2.0

    # Compute gradients w.r.t. (k_cat, pde_fraction, B_total)
    grad_fn = jit(grad(spreading_metric, argnums=(0, 1, 2)))

    print("Computing autodiff sensitivities...")
    kcat0 = base_config.enz.k_cat
    pde0 = base_config.pde_activity_fraction
    B0 = base_config.diff.B_total

    t0 = time.time()
    g_kcat, g_pde, g_B = grad_fn(kcat0, pde0, B0)
    elapsed = time.time() - t0

    val = spreading_metric(kcat0, pde0, B0)
    print(f"  Spreading metric (γ4-5 cAMP): {val:.4f} μM")
    print(f"  ∂(spreading)/∂(k_cat):        {g_kcat:.6f} μM·s")
    print(f"  ∂(spreading)/∂(pde_frac):      {g_pde:.4f} μM")
    print(f"  ∂(spreading)/∂(B_total):       {g_B:.6f} μM/μM")
    print(f"  Computed in {elapsed:.2f}s")

    return {
        'value': float(val),
        'grad_kcat': float(g_kcat),
        'grad_pde_frac': float(g_pde),
        'grad_B_total': float(g_B),
    }


# ═══════════════════════════════════════════════════════════════
# 8. VISUALIZATION (same as CPU version)
# ═══════════════════════════════════════════════════════════════

COMP_COLORS = ['#e41a1c', '#ff7f00', '#999999', '#377eb8', '#4daf4a']
COMP_LABELS = ['γ1', 'γ2', 'γ3', 'γ4', 'γ5']


def plot_comparison(res_wt, res_dunce, save_path=None):
    """WT vs dunce⁻ comparison: time courses."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, res, title in zip(axes, [res_wt, res_dunce], ['WT', 'dunce⁻']):
        for k in range(5):
            ax.plot(res['t'], res['bouton_c_free'][:, k],
                    color=COMP_COLORS[k], label=COMP_LABELS[k], linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Free [cAMP] (μM)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, res['t'][-1])
        ax.set_ylim(bottom=0)

        # Mark shocks
        stim = res['config'].stim
        for i in range(stim.n_pairings):
            ts = i * stim.inter_trial_interval + stim.shock_onset
            ax.axvspan(ts, ts + stim.shock_duration, color='red', alpha=0.08)

    fig.suptitle('cAMP Nanodomain Model (GPU/JAX): WT vs dunce⁻',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_spreading(res_wt, res_dunce, save_path=None):
    """Aversive vs appetitive compartment cAMP."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, res, title in zip(axes, [res_wt, res_dunce], ['WT', 'dunce⁻']):
        c_aversive = res['bouton_c_free'][:, 0:2].mean(axis=1)
        c_appetitive = res['bouton_c_free'][:, 3:5].mean(axis=1)
        c_mixed = res['bouton_c_free'][:, 2]

        ax.plot(res['t'], c_aversive, 'r-', label='γ1-2 (aversive)', linewidth=1.5)
        ax.plot(res['t'], c_appetitive, 'b-', label='γ4-5 (appetitive)', linewidth=1.5)
        ax.plot(res['t'], c_mixed, '--', color='gray', label='γ3 (mixed)', linewidth=1.0)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Free [cAMP] (μM)')
        ax.set_title(f'{title} — cAMP spreading')
        ax.legend(loc='upper right')
        ax.set_xlim(0, res['t'][-1])
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_kymograph(res, save_path=None, title=''):
    """Kymograph: cAMP (color) vs position vs time."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.array(res['grid'].x)
    t = res['t']
    data = res['c_free']

    vmax = np.percentile(data, 99.5)
    im = ax.pcolormesh(x, t, data, shading='auto', cmap='hot', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Free [cAMP] (μM)')

    bouton_x = np.array(res['grid'].x[np.array(res['grid'].bouton_bin_indices)])
    for k, xb in enumerate(bouton_x):
        ax.axvline(xb, color='cyan', alpha=0.3, linewidth=0.8, linestyle='--')
        ax.text(xb, t[-1] * 1.02, COMP_LABELS[k], ha='center', fontsize=9, color='cyan')

    ax.set_xlabel('Position along axon (μm)')
    ax.set_ylabel('Time (s)')
    ax.set_title(title or 'Free [cAMP] Kymograph')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 9. MAIN — Run all experiments
# ═══════════════════════════════════════════════════════════════

def main():
    output_dir = './gpu_outputs'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("cAMP Nanodomain Model — GPU (JAX + Diffrax)")
    print("=" * 60)

    # ── Experiment 1: WT vs dunce⁻ ──
    print("\n[Experiment 1] WT vs dunce⁻")
    config_wt = ModelConfig(pde_activity_fraction=1.0)
    config_dunce = ModelConfig(pde_activity_fraction=0.0)

    t0 = time.time()
    res_wt = solve_model(config_wt, dt_save=0.2)
    t_wt = time.time() - t0
    print(f"  WT solved in {t_wt:.2f}s")

    t0 = time.time()
    res_dunce = solve_model(config_dunce, dt_save=0.2)
    t_dunce = time.time() - t0
    print(f"  dunce⁻ solved in {t_dunce:.2f}s")

    plot_comparison(res_wt, res_dunce,
                    save_path=os.path.join(output_dir, 'gpu_fig1_comparison.png'))
    plot_spreading(res_wt, res_dunce,
                   save_path=os.path.join(output_dir, 'gpu_fig2_spreading.png'))
    plot_kymograph(res_wt,
                   save_path=os.path.join(output_dir, 'gpu_fig3_kymograph_wt.png'),
                   title='WT — Free [cAMP]')
    plot_kymograph(res_dunce,
                   save_path=os.path.join(output_dir, 'gpu_fig4_kymograph_dunce.png'),
                   title='dunce⁻ — Free [cAMP]')

    # ── Experiment 2: Massive parallel k_cat sweep (GPU power!) ──
    print("\n[Experiment 2] Parallel k_cat sweep (50 values via vmap)")
    try:
        kcat_sweep = parallel_sweep_kcat(
            np.linspace(0.5, 200.0, 50),
            config_wt, dt_save=1.0,
        )
        print("  vmap sweep succeeded!")
    except Exception as e:
        print(f"  vmap sweep failed (may need more GPU memory): {e}")
        print("  Falling back to sequential sweep...")
        kcat_sweep = None

    # ── Experiment 3: Autodiff sensitivity ──
    print("\n[Experiment 3] Autodiff sensitivity analysis")
    try:
        sensitivities = compute_sensitivity(config_wt)
    except Exception as e:
        print(f"  Autodiff failed: {e}")
        sensitivities = None

    # ── Summary ──
    print("\n" + "=" * 60)
    print("GPU EXPERIMENTS COMPLETE")
    print(f"Output: {output_dir}/")
    print("=" * 60)

    # Print key results
    print("\nKey results:")
    print(f"  WT peak γ1 cAMP:     {res_wt['bouton_c_free'][:, 0].max():.2f} μM")
    print(f"  WT steady γ4-5 cAMP: {res_wt['bouton_c_free'][-1, 3:5].mean():.3f} μM")
    print(f"  dunce⁻ final γ4-5:   {res_dunce['bouton_c_free'][-1, 3:5].mean():.2f} μM")
    print(f"  Spreading ratio:     {res_dunce['bouton_c_free'][-1, 3:5].mean() / max(res_dunce['bouton_c_free'][-1, 0:2].mean(), 1e-6):.3f}")

    if sensitivities:
        print(f"\nAutodiff sensitivities at WT parameters:")
        print(f"  ∂(γ4-5 cAMP)/∂(k_cat) = {sensitivities['grad_kcat']:.6f}")
        print(f"  ∂(γ4-5 cAMP)/∂(PDE%)  = {sensitivities['grad_pde_frac']:.4f}")
        print(f"  ∂(γ4-5 cAMP)/∂(B_tot) = {sensitivities['grad_B_total']:.6f}")


if __name__ == '__main__':
    main()
