#!/usr/bin/env python3
"""
cAMP Model v4 — Component Decomposition Diagnostic
====================================================

Runs the forward model with individual components enabled/disabled
to show what each piece contributes to the predicted signal.
Also identifies the structural problem with K_Ca vs free [Ca²⁺].

Usage:
    python diagnose_components.py --data all_camp_long.csv --params best_params.json
    python diagnose_components.py --data all_camp_long.csv   # uses defaults
"""

import os
import json
import argparse
import numpy as np

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import jax
import jax.numpy as jnp
from jax import vmap, lax
jax.config.update("jax_enable_x64", True)

import diffrax

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---- Import geometry and stimulation from v4 ----
# (Inline minimal versions to keep this self-contained)

def build_geometry(n_comp=5, d_bouton=2.0, d_axon=0.3,
                   inter_comp=25.0, axon_bins=40):
    A_b = np.pi * (d_bouton / 2) ** 2
    A_a = np.pi * (d_axon / 2) ** 2
    axon_len = inter_comp - d_bouton
    ax_dx = axon_len / axon_bins
    positions, dx_arr, areas, types = [], [], [], []
    bouton_indices = []
    cx, idx = 0.0, 0
    for i in range(n_comp):
        bouton_indices.append(idx)
        positions.append(cx + d_bouton / 2)
        dx_arr.append(d_bouton); areas.append(A_b); types.append(0); idx += 1
        cx += d_bouton
        if i < n_comp - 1:
            for j in range(axon_bins):
                positions.append(cx + (j + 0.5) * ax_dx)
                dx_arr.append(ax_dx); areas.append(A_a); types.append(1); idx += 1
            cx += axon_len
    x, dx, A = np.array(positions), np.array(dx_arr), np.array(areas)
    types_arr = np.array(types); N = idx
    alpha_p, alpha_m = np.zeros(N), np.zeros(N)
    for i in range(N - 1):
        A_int = 2.0 * A[i] * A[i+1] / (A[i] + A[i+1])
        dist = 0.5 * (dx[i] + dx[i+1])
        flux = A_int / dist
        alpha_p[i] = flux / (A[i] * dx[i])
        alpha_m[i+1] = flux / (A[i+1] * dx[i+1])
    bouton_mask = (types_arr == 0).astype(float)
    da_mask_per_comp = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
    da_mask = np.zeros(N)
    for i, bi in enumerate(bouton_indices):
        da_mask[bi] = da_mask_per_comp[i]
    return {
        'alpha_p': jnp.array(alpha_p), 'alpha_m': jnp.array(alpha_m),
        'bouton_mask': jnp.array(bouton_mask), 'stim_mask': jnp.array(bouton_mask),
        'da_mask': jnp.array(da_mask),
        'bouton_idx': jnp.array(bouton_indices, dtype=jnp.int32), 'n_total': N,
    }


def smooth_pulse(t, onset, dur, steep=20.0):
    return jax.nn.sigmoid(steep*(t-onset)) - jax.nn.sigmoid(steep*(t-onset-dur))

def make_stim(onsets, odor_dur=5.0, shock_off=4.0, shock_dur=1.0):
    ons = jnp.array(onsets)
    def odor(t): return jnp.clip(jnp.sum(vmap(lambda o: smooth_pulse(t,o,odor_dur))(ons)),0,1)
    def shock(t): return jnp.clip(jnp.sum(vmap(lambda o: smooth_pulse(t,o+shock_off,shock_dur))(ons)),0,1)
    return odor, shock

def sensor_lowpass(signal, dt, tau):
    a = 1.0 - jnp.exp(-dt / tau)
    def step(F, s):
        F2 = F + a*(s-F)
        return F2, F2
    _, filt = lax.scan(step, signal[0], signal[1:])
    return jnp.concatenate([signal[:1], filt], axis=0)


# =============================================================================
# DIAGNOSTIC 1: Ca²⁺ dynamics — what does free [Ca²⁺] actually look like?
# =============================================================================

def diagnose_calcium(pairing_onsets, save_dir='.'):
    """Show free [Ca²⁺] time courses for different Ca_odor/Ca_shock values."""
    print("\n" + "="*60)
    print("DIAGNOSTIC 1: Ca²⁺ dynamics with RBA")
    print("="*60)

    kappa_S = 77.0
    tau_Ca = 0.046   # 46 ms
    Ca_rest = 0.05   # 50 nM
    K_Ca = 0.5       # current fixed value

    rba = 1.0 / (1.0 + kappa_S)
    print(f"  κ_S = {kappa_S}, τ_Ca = {tau_Ca*1000:.0f} ms, RBA factor = {rba:.4f}")
    print(f"  K_Ca = {K_Ca} μM (FIXED — this is the problem!)")

    dt = 0.001  # 1 ms resolution
    t = np.arange(-2.0, 15.0, dt)
    odor_fn, shock_fn = make_stim(pairing_onsets[:3])  # just 3 pairings

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel A: Free [Ca²⁺] for different Ca_shock values
    ax = axes[0]
    Ca_odor_val = 10.0
    for Ca_shock_val, color in [(1, 'blue'), (10, 'green'), (50, 'orange'), (200, 'red')]:
        ca = np.full_like(t, Ca_rest)
        for i in range(1, len(t)):
            odor_on = float(odor_fn(t[i]))
            shock_on = float(shock_fn(t[i]))
            j_total = Ca_odor_val * odor_on + Ca_shock_val * shock_on
            j_free = j_total * rba
            ca[i] = ca[i-1] + dt * (-(ca[i-1] - Ca_rest)/tau_Ca + j_free)
        ax.plot(t, ca*1000, color=color, label=f'Ca_shock={Ca_shock_val}')
    ax.axhline(K_Ca*1000, color='black', linestyle='--', linewidth=2, label=f'K_Ca = {K_Ca*1000:.0f} nM')
    ax.set_ylabel('Free [Ca²⁺] (nM)')
    ax.set_title(f'Free [Ca²⁺] with RBA (κ_S={kappa_S:.0f}), Ca_odor={Ca_odor_val}')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(600, K_Ca*1200))

    # Panel B: f(Ca) = Hill function value
    ax = axes[1]
    Ca_odor_val = 10.0
    for Ca_shock_val, color in [(1, 'blue'), (10, 'green'), (50, 'orange'), (200, 'red')]:
        ca = np.full_like(t, Ca_rest)
        for i in range(1, len(t)):
            odor_on = float(odor_fn(t[i]))
            shock_on = float(shock_fn(t[i]))
            j_free = (Ca_odor_val * odor_on + Ca_shock_val * shock_on) * rba
            ca[i] = ca[i-1] + dt * (-(ca[i-1] - Ca_rest)/tau_Ca + j_free)
        n_hill = 2.0
        f_ca = ca**n_hill / (K_Ca**n_hill + ca**n_hill)
        ax.plot(t, f_ca, color=color, label=f'Ca_shock={Ca_shock_val}, peak f(Ca)={f_ca.max():.3f}')
    ax.set_ylabel('f(Ca) = Ca²/(K_Ca² + Ca²)')
    ax.set_title(f'AC activation fraction — K_Ca = {K_Ca} μM')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # Panel C: f(Ca) with LOWERED K_Ca
    ax = axes[2]
    K_Ca_low = 0.08  # 80 nM — near resting [Ca²⁺]
    Ca_odor_val = 10.0
    for Ca_shock_val, color in [(1, 'blue'), (10, 'green'), (50, 'orange'), (200, 'red')]:
        ca = np.full_like(t, Ca_rest)
        for i in range(1, len(t)):
            odor_on = float(odor_fn(t[i]))
            shock_on = float(shock_fn(t[i]))
            j_free = (Ca_odor_val * odor_on + Ca_shock_val * shock_on) * rba
            ca[i] = ca[i-1] + dt * (-(ca[i-1] - Ca_rest)/tau_Ca + j_free)
        n_hill = 2.0
        f_ca = ca**n_hill / (K_Ca_low**n_hill + ca**n_hill)
        ax.plot(t, f_ca, color=color, label=f'Ca_shock={Ca_shock_val}, peak f(Ca)={f_ca.max():.3f}')
    ax.set_ylabel('f(Ca)')
    ax.set_title(f'AC activation with K_Ca = {K_Ca_low} μM (lowered to match free [Ca²⁺] range)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Time (s)')

    # Add shock markers
    for ax in axes:
        for onset in pairing_onsets[:3]:
            ax.axvline(onset + 4.0, color='red', alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(save_dir, 'diag1_calcium_dynamics.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print the key numbers
    print(f"\n  THE PROBLEM:")
    print(f"  With K_Ca = {K_Ca} μM and RBA (κ_S = 77):")
    print(f"    Resting free [Ca²⁺] = {Ca_rest*1000:.0f} nM")
    print(f"    K_Ca = {K_Ca*1000:.0f} nM")
    print(f"    → Resting f(Ca) = {(Ca_rest**2/(K_Ca**2+Ca_rest**2)):.4f}")
    print(f"    → Even Ca_shock=200 gives peak free [Ca²⁺] ≈ {Ca_rest*1000 + 200/78*tau_Ca*1000:.0f} nM")
    print(f"    → Peak f(Ca) ≈ {((Ca_rest+200/78*tau_Ca)**2/(K_Ca**2+(Ca_rest+200/78*tau_Ca)**2)):.3f}")
    print(f"")
    print(f"  With K_Ca = {K_Ca_low} μM (lowered):")
    print(f"    → Resting f(Ca) = {(Ca_rest**2/(K_Ca_low**2+Ca_rest**2)):.3f}")
    print(f"    → Ca_shock=50 peak f(Ca) ≈ {((Ca_rest+50/78*tau_Ca)**2/(K_Ca_low**2+(Ca_rest+50/78*tau_Ca)**2)):.3f}")
    print(f"")
    print(f"  CONCLUSION: K_Ca must be a FREE parameter or set much lower.")
    print(f"  The current K_Ca = 0.5 μM is appropriate for TOTAL [Ca²⁺],")
    print(f"  but after RBA, free [Ca²⁺] only reaches ~50-200 nM.")
    print(f"  The Hill function is essentially zero → optimizer ignores stimulus.")


# =============================================================================
# DIAGNOSTIC 2: Component-by-component forward model
# =============================================================================

def diagnose_components(params_dict, pairing_onsets, save_dir='.'):
    """Run model with individual components toggled on/off."""
    print("\n" + "="*60)
    print("DIAGNOSTIC 2: Component decomposition")
    print("="*60)

    geom = build_geometry()
    N = geom['n_total']
    t_save = jnp.arange(-5.0, 82.0, 0.5)

    # Build parameter vector
    param_names = ['D_free','B_total','K_D_buffer','k_on_buffer','V_basal',
                   'V_Ca','K_Ca','n_Hill_AC','k_cat','K_m','PDE_conc',
                   'Ca_rest','Ca_odor','tau_Ca','kappa_S','alpha','dff_offset',
                   'tau_sensor','V_coinc','Ca_shock']
    base_params = jnp.array([params_dict.get(n, 0.0) for n in param_names])

    scenarios = {
        'A: V_basal only\n(no stimulus)': {
            'V_Ca': 0.0, 'V_coinc': 0.0, 'Ca_odor': 0.0, 'Ca_shock': 0.0,
        },
        'B: V_basal + V_Ca\n(odour Ca²⁺ only)': {
            'V_coinc': 0.0, 'Ca_shock': 0.0,
        },
        'C: V_basal + V_coinc\n(DA coincidence only)': {
            'V_Ca': 0.0, 'Ca_odor': 0.0,
            # Need some Ca²⁺ for f(Ca) to be nonzero during shock
            'Ca_shock': params_dict.get('Ca_shock', 20.0),
        },
        'D: Full model\n(all terms)': {},
        'E: Full model\n(K_Ca=0.08 μM)': {
            'K_Ca': 0.08,
        },
        'F: Biologically tuned\n(K_Ca=0.08, V_basal=0.1)': {
            'K_Ca': 0.08, 'V_basal': 0.1,
            'V_Ca': 5.0, 'V_coinc': 50.0,
            'Ca_odor': 10.0, 'Ca_shock': 80.0,
            'alpha': 0.05, 'B_total': 20.0,
        },
    }

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(16, 4*len(scenarios)),
                              sharex=True, sharey=True)
    colors = ['#e41a1c', '#ff7f00', '#999999', '#377eb8', '#4daf4a']

    for row, (label, overrides) in enumerate(scenarios.items()):
        # Build modified params
        p = np.array(base_params)
        for name, val in overrides.items():
            idx = param_names.index(name)
            p[idx] = val
        params = jnp.array(p)

        for col, (geno, is_dunce) in enumerate([('WT', 0.0), ('dunce-KD', 1.0)]):
            ax = axes[row, col]

            # Run forward model
            odor_fn, shock_fn = make_stim(pairing_onsets)
            term = make_ode_term_simple(geom, params, odor_fn, shock_fn, is_dunce, param_names)

            y0 = jnp.zeros(3 * N)
            y0 = y0.at[2*N:3*N].set(params[param_names.index('Ca_rest')])

            solver = diffrax.Kvaerno5()
            sc = diffrax.PIDController(rtol=1e-5, atol=1e-7, dtmin=1e-6, dtmax=0.5)
            sol = diffrax.diffeqsolve(
                term, solver, t0=t_save[0], t1=t_save[-1], dt0=0.01,
                y0=y0, saveat=diffrax.SaveAt(ts=t_save),
                stepsize_controller=sc, max_steps=200000, throw=False)

            c_free = sol.ys[:, :N]
            bouton_c = c_free[:, geom['bouton_idx']]
            alpha_val = params[param_names.index('alpha')]
            offset_val = params[param_names.index('dff_offset')]
            tau_s = params[param_names.index('tau_sensor')]
            raw = alpha_val * bouton_c + offset_val
            pred = sensor_lowpass(raw, 0.5, tau_s)

            for ci in range(5):
                ax.plot(np.array(t_save), np.array(pred[:, ci]),
                       color=colors[ci], linewidth=1.5, label=f'γ{ci+1}')

            ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
            ax.set_ylim(-0.3, 1.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight='bold')
            if row == 0:
                ax.set_title(geno, fontsize=12, fontweight='bold')
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc='upper right')
            for o in pairing_onsets:
                ax.axvline(o, color='red', alpha=0.1, linewidth=0.5)

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    fig.suptitle('Component Decomposition: What does each model term contribute?',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, 'diag2_component_decomposition.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def make_ode_term_simple(geom, params, odor_fn, shock_fn, is_dunce, param_names):
    """Simplified ODE term for diagnostics (matches v4 physics)."""
    N = geom['n_total']
    idx = {n: i for i, n in enumerate(param_names)}

    D = params[idx['D_free']]; B_tot = params[idx['B_total']]
    K_D = params[idx['K_D_buffer']]; k_on = params[idx['k_on_buffer']]
    V_bas = params[idx['V_basal']]; V_ca = params[idx['V_Ca']]
    K_Ca = params[idx['K_Ca']]; nH = params[idx['n_Hill_AC']]
    kcat = params[idx['k_cat']]; Km = params[idx['K_m']]
    PDE_c = params[idx['PDE_conc']]; Ca_r = params[idx['Ca_rest']]
    Ca_od = params[idx['Ca_odor']]; tau_ca = params[idx['tau_Ca']]
    kS = params[idx['kappa_S']]; V_co = params[idx['V_coinc']]
    Ca_sh = params[idx['Ca_shock']]

    k_off = k_on * K_D; Vpde = PDE_c * kcat
    rba = 1.0 / (1.0 + kS)
    pde_m = geom['bouton_mask'] * (1.0 - is_dunce)
    bm = geom['bouton_mask']; sm = geom['stim_mask']; dm = geom['da_mask']
    ap = geom['alpha_p']; am = geom['alpha_m']

    def vf(t, y, args):
        c, cb, ca = y[:N], y[N:2*N], y[2*N:3*N]
        c_p, cb_p = jnp.maximum(c, 0.0), jnp.maximum(cb, 0.0)

        dc = jnp.zeros(N)
        dc = dc.at[:-1].add(D * ap[:-1] * (c[1:] - c[:-1]))
        dc = dc.at[1:].add(D * am[1:] * (c[:-1] - c[1:]))

        f_ca = ca**nH / (K_Ca**nH + ca**nH + 1e-30)
        od = odor_fn(t); sh = shock_fn(t)
        dc = dc + bm*V_bas + bm*V_ca*f_ca*od + dm*V_co*f_ca*sh
        dc = dc - pde_m * Vpde * c_p / (Km + c_p + 1e-30)

        Bf = jnp.maximum(B_tot - cb_p, 0.0)
        jb = k_on*c_p*Bf; ju = k_off*cb_p
        dc = dc - jb + ju

        j_ca = sm * (Ca_od*od + Ca_sh*sh) * rba
        dca = -(ca - Ca_r)/tau_ca + j_ca

        return jnp.concatenate([dc, jb - ju, dca])

    return diffrax.ODETerm(vf)


# =============================================================================
# DIAGNOSTIC 3: K_Ca sensitivity analysis
# =============================================================================

def diagnose_kca_sensitivity(params_dict, pairing_onsets, save_dir='.'):
    """Show how K_Ca value affects the model output."""
    print("\n" + "="*60)
    print("DIAGNOSTIC 3: K_Ca sensitivity")
    print("="*60)

    geom = build_geometry()
    N = geom['n_total']
    t_save = jnp.arange(-5.0, 82.0, 0.5)

    param_names = ['D_free','B_total','K_D_buffer','k_on_buffer','V_basal',
                   'V_Ca','K_Ca','n_Hill_AC','k_cat','K_m','PDE_conc',
                   'Ca_rest','Ca_odor','tau_Ca','kappa_S','alpha','dff_offset',
                   'tau_sensor','V_coinc','Ca_shock']

    # Use biologically informed values (not optimizer's lazy solution)
    bio_params = {
        'D_free': 130.0, 'B_total': 20.0, 'K_D_buffer': 2.0, 'k_on_buffer': 10.0,
        'V_basal': 0.1,     # LOW: basal should be small
        'V_Ca': 5.0,        # moderate Ca²⁺-only activation
        'K_Ca': 0.5,        # will be varied
        'n_Hill_AC': 2.0,
        'k_cat': 20.0, 'K_m': 2.4, 'PDE_conc': 2.0,
        'Ca_rest': 0.05, 'Ca_odor': 10.0, 'tau_Ca': 0.046,
        'kappa_S': 77.0,
        'alpha': 0.05, 'dff_offset': 0.0, 'tau_sensor': 3.0,
        'V_coinc': 50.0, 'Ca_shock': 80.0,
    }

    K_Ca_values = [0.5, 0.2, 0.1, 0.08, 0.06, 0.04]

    fig, axes = plt.subplots(len(K_Ca_values), 2, figsize=(16, 4*len(K_Ca_values)),
                              sharex=True, sharey=True)
    colors = ['#e41a1c', '#ff7f00', '#999999', '#377eb8', '#4daf4a']

    for row, K_Ca_val in enumerate(K_Ca_values):
        bio_params['K_Ca'] = K_Ca_val
        p = jnp.array([bio_params[n] for n in param_names])

        for col, (geno, is_d) in enumerate([('WT', 0.0), ('dunce-KD', 1.0)]):
            ax = axes[row, col]
            odor_fn, shock_fn = make_stim(pairing_onsets)
            term = make_ode_term_simple(geom, p, odor_fn, shock_fn, is_d, param_names)
            y0 = jnp.zeros(3*N).at[2*N:3*N].set(0.05)
            sol = diffrax.diffeqsolve(
                term, diffrax.Kvaerno5(), t0=t_save[0], t1=t_save[-1], dt0=0.01,
                y0=y0, saveat=diffrax.SaveAt(ts=t_save),
                stepsize_controller=diffrax.PIDController(rtol=1e-5,atol=1e-7,dtmin=1e-6,dtmax=0.5),
                max_steps=200000, throw=False)

            bouton_c = sol.ys[:, :N][:, geom['bouton_idx']]
            raw = 0.05 * bouton_c + 0.0
            pred = sensor_lowpass(raw, 0.5, 3.0)

            for ci in range(5):
                ax.plot(np.array(t_save), np.array(pred[:, ci]),
                       color=colors[ci], linewidth=1.5, label=f'γ{ci+1}')

            ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
            ax.set_ylim(-0.5, 3.0)
            if col == 0:
                ax.set_ylabel(f'K_Ca = {K_Ca_val} μM', fontsize=10, fontweight='bold')
            if row == 0:
                ax.set_title(geno, fontsize=12, fontweight='bold')
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc='upper right')
            for o in pairing_onsets:
                ax.axvline(o, color='red', alpha=0.1, linewidth=0.5)

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    fig.suptitle('K_Ca Sensitivity: lowering K_Ca activates the stimulus pathway',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, 'diag3_kca_sensitivity.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print summary table
    print(f"\n  K_Ca sensitivity (peak free [Ca²⁺] during shock ≈ 170 nM with Ca_shock=80):")
    print(f"  {'K_Ca (μM)':>10s} {'K_Ca (nM)':>10s} {'peak f(Ca)':>12s} {'Status':>20s}")
    print(f"  {'-'*55}")
    ca_peak = 0.05 + 80.0/78.0 * 0.046  # approximate peak free [Ca²⁺]
    for kca in K_Ca_values:
        fca = ca_peak**2 / (kca**2 + ca_peak**2)
        status = "← DEAD ZONE" if fca < 0.05 else ("← ACTIVE" if fca > 0.2 else "← MARGINAL")
        print(f"  {kca:10.3f} {kca*1000:10.0f} {fca:12.4f} {status:>20s}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Model component diagnostics')
    parser.add_argument('--params', type=str, default=None)
    parser.add_argument('--outdir', type=str, default='diagnostics')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pairing_onsets = np.arange(0.0, 60.0, 5.0).tolist()

    # Load params if available
    if args.params and os.path.exists(args.params):
        with open(args.params) as f:
            saved = json.load(f)
        params_dict = saved['all_params']
        print(f"Loaded parameters from: {args.params}")
    else:
        # Use v4 defaults
        params_dict = {
            'D_free': 130.0, 'B_total': 20.0, 'K_D_buffer': 2.0, 'k_on_buffer': 10.0,
            'V_basal': 0.5, 'V_Ca': 2.0, 'K_Ca': 0.5, 'n_Hill_AC': 2.0,
            'k_cat': 5.0, 'K_m': 2.4, 'PDE_conc': 1.0,
            'Ca_rest': 0.05, 'Ca_odor': 5.0, 'tau_Ca': 0.046, 'kappa_S': 77.0,
            'alpha': 0.1, 'dff_offset': 0.0, 'tau_sensor': 3.0,
            'V_coinc': 15.0, 'Ca_shock': 20.0,
        }
        print("Using default parameters")

    # Run diagnostics
    diagnose_calcium(pairing_onsets, save_dir=args.outdir)
    diagnose_components(params_dict, pairing_onsets, save_dir=args.outdir)
    diagnose_kca_sensitivity(params_dict, pairing_onsets, save_dir=args.outdir)

    print(f"\n{'='*60}")
    print("SUMMARY: Root cause of optimizer failure")
    print("="*60)
    print("""
  The optimizer shuts off the stimulus pathway because:

  1. K_Ca = 0.5 μM is FIXED and too high for RBA-buffered free [Ca²⁺]
  2. With κ_S = 77, free [Ca²⁺] only reaches ~50-200 nM
  3. The Hill function f(Ca) = Ca²/(K_Ca² + Ca²) ≈ 0.01 at these levels
  4. So V_Ca × f(Ca) ≈ 0 and V_coinc × f(Ca) ≈ 0 regardless of bounds
  5. The optimizer compensates by cranking V_basal to ~4 μM/s

  FIX: Make K_Ca a FREE parameter with bounds [0.03, 0.5] μM.
  This lets the optimizer find a K_Ca that matches the RBA-scaled Ca²⁺.

  Alternatively: express the Hill function in terms of TOTAL [Ca²⁺]
  (free + bound = (1+κ_S) × free) instead of free [Ca²⁺]. Then
  K_Ca = 0.5 μM would correspond to total influx, not free.

  The biophysical justification: calmodulin competes with endogenous
  buffers for Ca²⁺ binding. The effective K_Ca for rutabaga activation
  depends on the local buffer environment, so it should be fitted.
""")

    print(f"All diagnostics in: {args.outdir}/")


if __name__ == '__main__':
    main()
