#!/usr/bin/env python3
"""
Plot Figure 2-style panels from experimental data.
Optionally overlay model fit from best_params.json.

Usage:
    python plot_figure2.py --data all_camp_long.csv
    python plot_figure2.py --data all_camp_long.csv --params best_params.json
    python plot_figure2.py --data all_camp_long.csv --params best_params.json --t-range -15,90
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Colors matching your Figure 2 ──
WT_COLOR  = '#4daf4a'
KD_COLOR  = '#ff7f00'
WT_FILL   = '#a6d96a'
KD_FILL   = '#fdae61'


# =============================================================================
# DATA
# =============================================================================

def load_data(csv_path):
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


# =============================================================================
# CPU MODEL (with sensor filter)
# =============================================================================

def sensor_filter_cpu(signal, dt, tau_sensor):
    """Exponential low-pass: dF/dt = (signal - F) / tau_sensor"""
    alpha = 1.0 - np.exp(-dt / tau_sensor)
    F = np.zeros_like(signal)
    F[0] = signal[0]
    for i in range(1, len(signal)):
        F[i] = F[i-1] + alpha * (signal[i] - F[i-1])
    return F


def run_cpu_model(params, pairing_onsets, is_dunce, t_eval):
    """Minimal CPU model run. Returns predicted ΔF/F at 5 boutons.

    Uses scipy.integrate.solve_ivp. Includes sensor filter.
    """
    # Geometry — realistic Drosophila MB γ-lobe dimensions
    # Aso et al. 2014, Scheffer et al. 2020: γ-lobe ~100-125 μm total
    n_comp = 5
    d_b, d_a = 2.0, 0.3          # bouton and axon diameter (μm)
    inter_comp = 25.0             # center-to-center between compartments (μm)
    n_ax = 40                     # spatial bins per axon segment
    A_b = np.pi * (d_b/2)**2
    A_a = np.pi * (d_a/2)**2
    ax_len = inter_comp - d_b
    ax_dx = ax_len / n_ax

    # Build grid
    dx_arr, A_arr, btypes = [], [], []
    bouton_idx = []
    idx = 0
    for i in range(n_comp):
        bouton_idx.append(idx)
        dx_arr.append(d_b); A_arr.append(A_b); btypes.append(True)
        idx += 1
        if i < n_comp - 1:
            for _ in range(n_ax):
                dx_arr.append(ax_dx); A_arr.append(A_a); btypes.append(False)
                idx += 1
    N = idx
    dx_arr = np.array(dx_arr)
    A_arr = np.array(A_arr)
    is_bouton = np.array(btypes)

    # Coupling coefficients
    alpha_p = np.zeros(N)
    alpha_m = np.zeros(N)
    for i in range(N-1):
        A_int = 2*A_arr[i]*A_arr[i+1]/(A_arr[i]+A_arr[i+1])
        dist = 0.5*(dx_arr[i]+dx_arr[i+1])
        flux = A_int/dist
        V_i = A_arr[i]*dx_arr[i]
        V_ip1 = A_arr[i+1]*dx_arr[i+1]
        alpha_p[i] = flux/V_i
        alpha_m[i+1] = flux/V_ip1

    # Unpack params
    D       = params['D_free']
    B_tot   = params['B_total']
    K_D     = params['K_D_buffer']
    k_on    = params['k_on_buffer']
    k_off   = k_on * K_D
    V_bas   = params['V_basal']
    V_ac    = params['V_max_AC']
    K_Ca    = params['K_Ca']
    n_H     = params['n_Hill_AC']
    k_cat   = params['k_cat']
    K_m     = params['K_m']
    PDE_c   = params['PDE_conc']
    V_pde   = PDE_c * k_cat
    Ca_rest = params['Ca_rest']
    Ca_amp  = params['Ca_amplitude']
    tau_Ca  = params['tau_Ca']
    Ca_dur  = params['Ca_pulse_dur']
    alpha_s = params['alpha']
    offset  = params['dff_offset']
    tau_sen = params['tau_sensor']

    odor_dur = 5.0
    shock_off = 4.0
    shock_dur = 1.0
    pde_mask = is_bouton * (0.0 if is_dunce else 1.0)
    stim_mask = is_bouton.astype(float)  # ALL boutons receive Ca²⁺

    def rhs(t, y):
        c = y[:N]; cb = y[N:2*N]; ca = y[2*N:3*N]
        c_pos = np.maximum(c, 0.0)
        cb_pos = np.maximum(cb, 0.0)

        dc = np.zeros(N)
        dc[:-1] += D * alpha_p[:-1] * (c[1:] - c[:-1])
        dc[1:]  += D * alpha_m[1:]  * (c[:-1] - c[1:])

        odor_on = any(o <= t < o + odor_dur for o in pairing_onsets)
        for bi in bouton_idx:
            dc[bi] += V_bas
            if odor_on:
                f_ca = ca[bi]**n_H / (K_Ca**n_H + ca[bi]**n_H + 1e-30)
                dc[bi] += V_ac * f_ca

        dc -= pde_mask * V_pde * c_pos / (K_m + c_pos + 1e-30)

        B_free = np.maximum(B_tot - cb_pos, 0.0)
        j_bind = k_on * c_pos * B_free
        j_unbind = k_off * cb_pos
        dc  += -j_bind + j_unbind
        dcb  =  j_bind - j_unbind

        dca = -(ca - Ca_rest) / tau_Ca
        shock_on = any(o + shock_off <= t < o + shock_off + shock_dur
                       for o in pairing_onsets)
        if shock_on:
            dca += stim_mask * Ca_amp / Ca_dur

        return np.concatenate([dc, dcb, dca])

    y0 = np.zeros(3*N)
    y0[2*N:3*N] = Ca_rest

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0,
                    method='BDF', t_eval=t_eval,
                    max_step=0.1, rtol=1e-6, atol=1e-8)

    bouton_c = np.column_stack([sol.y[bi, :] for bi in bouton_idx])  # [n_t, 5]
    raw = alpha_s * bouton_c + offset

    # Sensor filter
    dt = t_eval[1] - t_eval[0]
    filtered = np.zeros_like(raw)
    for k in range(5):
        filtered[:, k] = sensor_filter_cpu(raw[:, k], dt, tau_sen)

    return sol.t, filtered


# =============================================================================
# FIGURE 2 PLOT
# =============================================================================

def plot_figure2(data, params_dict=None, pairing_onsets=None,
                 t_range=(-15, 90), outdir='.', tag=''):
    """Figure 2 left-column style: 5 panels (γ1→γ5), mean±SEM, model overlay."""
    comps = ['g1', 'g2', 'g3', 'g4', 'g5']
    t_lo, t_hi = t_range

    # Run model if params provided
    model_traces = {}
    if params_dict is not None and pairing_onsets is not None:
        t_model = np.arange(max(t_lo, -10), min(t_hi + 5, 85), 0.25)
        for geno, is_d in [('dnc-wt', False), ('dnc-KD', True)]:
            print(f"  Running {geno} model...")
            t_m, pred = run_cpu_model(params_dict, pairing_onsets, is_d, t_model)
            model_traces[geno] = (t_m, pred)
        print("  Model runs complete.")

    fig, axes = plt.subplots(5, 1, figsize=(7, 16), sharex=True)

    for row, comp in enumerate(comps):
        ax = axes[row]
        ci = row

        for geno, color, fill, label in [
            ('dnc-wt', WT_COLOR, WT_FILL, 'dnc-wt'),
            ('dnc-KD', KD_COLOR, KD_FILL, 'dnc-KD'),
        ]:
            key = (geno, comp)
            if key in data:
                t_d, m_d, s_d = data[key]
                mask = (t_d >= t_lo) & (t_d <= t_hi)
                ax.fill_between(t_d[mask], m_d[mask]-s_d[mask],
                                m_d[mask]+s_d[mask],
                                alpha=0.25, color=fill, linewidth=0)
                ax.plot(t_d[mask], m_d[mask], '-', color=color,
                       linewidth=1.5, label=label)

            # Model overlay
            if geno in model_traces:
                t_m, pred = model_traces[geno]
                m_mask = (t_m >= t_lo) & (t_m <= t_hi)
                ax.plot(t_m[m_mask], pred[m_mask, ci], '--', color=color,
                       linewidth=2.0, alpha=0.7)

        # Stimulus triangles
        if pairing_onsets is not None:
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

    for ext in ['png', 'pdf']:
        path = os.path.join(outdir, f'figure2_model{tag}.{ext}')
        fig.savefig(path, dpi=250 if ext == 'png' else None,
                    bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Figure 2-style plot')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--params', type=str, default=None,
                        help='best_params.json to overlay model fit')
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--t-range', type=str, default='-15,90',
                        help='Time display range (lo,hi)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_range = tuple(float(x) for x in args.t_range.split(','))

    print(f"Loading data: {args.data}")
    data = load_data(args.data)

    params_dict = None
    pairing_onsets = None
    if args.params:
        print(f"Loading params: {args.params}")
        with open(args.params) as f:
            saved = json.load(f)
        params_dict = saved['all_params']
        # Add tau_sensor if missing (backward compatibility with v1)
        if 'tau_sensor' not in params_dict:
            params_dict['tau_sensor'] = 5.0
            print("  NOTE: tau_sensor not in params, using default 5.0s")
        pairing_onsets = np.array(saved.get('stim_onsets',
                                            np.arange(0, 60, 5).tolist()))

    plot_figure2(data, params_dict, pairing_onsets,
                 t_range=t_range, outdir=args.outdir)


if __name__ == '__main__':
    main()
