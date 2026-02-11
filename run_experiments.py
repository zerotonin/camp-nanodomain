#!/usr/bin/env python3
"""
Run cAMP nanodomain model experiments.

Generates:
1. WT vs dunce⁻ comparison (kymographs + time courses)
2. Spatial profiles at key time points
3. Cumulative spreading analysis
4. Parameter sensitivity sweep (k_cat, D_eff, B_total)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camp_model import (
    ModelParams, default_params, dunce_mutant_params,
    run_simulation,
    plot_comparison, plot_spatial_comparison,
    plot_cumulative_spreading, plot_bouton_timecourses,
)

OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_wt_vs_dunce():
    """Experiment 1: WT vs dunce⁻ comparison."""
    print("=" * 60)
    print("EXPERIMENT 1: WT vs dunce⁻")
    print("=" * 60)

    # Wild-type
    params_wt = default_params()
    result_wt = run_simulation(params_wt, max_step=0.05, dense_output_dt=0.2)

    # dunce⁻ mutant
    params_dunce = dunce_mutant_params()
    result_dunce = run_simulation(params_dunce, max_step=0.05, dense_output_dt=0.2)

    # --- Figure 1: Main comparison ---
    fig1 = plot_comparison(result_wt, result_dunce,
                           save_path=os.path.join(OUTPUT_DIR, 'fig1_wt_vs_dunce.png'))
    plt.close(fig1)

    # --- Figure 2: Spatial profiles ---
    stim = params_wt.stimulation
    # Key time points: after 1st shock, after 3rd, after last, 10s after last
    time_points = [
        stim.shock_onset + stim.shock_duration,                          # after 1st shock
        2 * stim.inter_trial_interval + stim.shock_onset + stim.shock_duration,  # after 3rd
        (stim.n_pairings - 1) * stim.inter_trial_interval + stim.shock_onset + stim.shock_duration,  # after last
        (stim.n_pairings - 1) * stim.inter_trial_interval + stim.shock_onset + 10.0,  # 10s after last
    ]
    fig2 = plot_spatial_comparison(result_wt, result_dunce, time_points,
                                   save_path=os.path.join(OUTPUT_DIR, 'fig2_spatial_profiles.png'))
    plt.close(fig2)

    # --- Figure 3: Cumulative spreading ---
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_cumulative_spreading(result_wt, ax=axes[0], title='WT — cAMP spreading')
    plot_cumulative_spreading(result_dunce, ax=axes[1], title='dunce⁻ — cAMP spreading')
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'fig3_spreading.png'), dpi=200, bbox_inches='tight')
    print(f"Saved spreading figure")
    plt.close(fig3)

    return result_wt, result_dunce


def run_kcat_sensitivity():
    """Experiment 2: Sensitivity to PDE catalytic rate."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: k_cat sensitivity")
    print("=" * 60)

    kcat_values = [1.0, 5.0, 20.0, 50.0, 160.0]
    fig, axes = plt.subplots(1, len(kcat_values), figsize=(4 * len(kcat_values), 5),
                             sharey=True)

    for i, kcat in enumerate(kcat_values):
        params = default_params()
        params.enzyme.k_cat = kcat
        result = run_simulation(params, max_step=0.05, dense_output_dt=0.5)

        plot_bouton_timecourses(result, ax=axes[i],
                                title=f'k_cat = {kcat} s⁻¹')
        if i > 0:
            axes[i].set_ylabel('')

    fig.suptitle('WT: Sensitivity to dunce k_cat', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_kcat_sensitivity.png'),
                dpi=200, bbox_inches='tight')
    print("Saved k_cat sensitivity figure")
    plt.close(fig)


def run_buffering_sensitivity():
    """Experiment 3: Sensitivity to cAMP buffering capacity."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Buffer capacity sensitivity (dunce⁻)")
    print("=" * 60)

    btotal_values = [0.0, 5.0, 20.0, 50.0]

    fig, axes = plt.subplots(1, len(btotal_values),
                             figsize=(4 * len(btotal_values), 5), sharey=True)

    for i, btotal in enumerate(btotal_values):
        params = dunce_mutant_params()
        params.diffusion.B_total = btotal
        result = run_simulation(params, max_step=0.05, dense_output_dt=0.5)

        plot_bouton_timecourses(result, ax=axes[i],
                                title=f'B_total = {btotal} μM')
        if i > 0:
            axes[i].set_ylabel('')

    fig.suptitle('dunce⁻: Sensitivity to cAMP Buffer Capacity',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_buffer_sensitivity.png'),
                dpi=200, bbox_inches='tight')
    print("Saved buffer sensitivity figure")
    plt.close(fig)


def run_pde_dose_response():
    """Experiment 4: Partial PDE inhibition dose-response."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: PDE inhibition dose-response")
    print("=" * 60)

    inhibition_levels = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    stim_params = default_params().stimulation

    # Measure cAMP in γ4-5 at end of training
    t_measure = (
        (stim_params.n_pairings - 1) * stim_params.inter_trial_interval
        + stim_params.shock_onset + 5.0
    )

    spreading_metric = []

    for inhib in inhibition_levels:
        params = default_params()
        params.pde_inhibition = inhib
        result = run_simulation(params, max_step=0.05, dense_output_dt=0.5)

        # Get cAMP in γ4-5 at measurement time
        t_idx = np.argmin(np.abs(result.t - t_measure))
        c_appetitive = result.bouton_c_free[t_idx, 3:5].mean()
        c_aversive = result.bouton_c_free[t_idx, 0:2].mean()

        ratio = c_appetitive / max(c_aversive, 1e-6)
        spreading_metric.append(ratio)
        print(f"  PDE inhib = {inhib*100:.0f}%: γ4-5 = {c_appetitive:.3f} μM, "
              f"γ1-2 = {c_aversive:.3f} μM, ratio = {ratio:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.array(inhibition_levels) * 100, spreading_metric,
            'ko-', linewidth=2, markersize=8)
    ax.set_xlabel('PDE Inhibition (%)')
    ax.set_ylabel('cAMP spreading ratio (γ4-5 / γ1-2)')
    ax.set_title('PDE Dose-Response: cAMP Spreading')
    ax.set_xlim(-5, 105)
    ax.set_ylim(bottom=0)
    ax.axhline(spreading_metric[0], color='green', linestyle='--',
               alpha=0.5, label='WT level')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_pde_dose_response.png'),
                dpi=200, bbox_inches='tight')
    print("Saved PDE dose-response figure")
    plt.close(fig)


def print_model_summary(params: ModelParams):
    """Print key model parameters and derived quantities."""
    from camp_model.geometry import build_grid
    grid = build_grid(params.geometry)

    print("\n" + "=" * 60)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 60)
    print(f"\nGeometry:")
    print(f"  Bouton diameter:     {params.geometry.bouton_diameter} μm")
    print(f"  Axon diameter:       {params.geometry.axon_diameter} μm")
    print(f"  Inter-bouton dist:   {params.geometry.inter_bouton_distance} μm")
    print(f"  Bouton volume:       {params.geometry.bouton_volume:.3f} μm³")
    print(f"  Axon cross-section:  {params.geometry.axon_cross_section:.4f} μm²")
    print(f"  Grid points:         {grid.n_total}")

    print(f"\nDiffusion:")
    print(f"  D_free:              {params.diffusion.D_free} μm²/s")
    print(f"  D_eff (low cAMP):    {params.diffusion.D_eff_low:.1f} μm²/s")
    print(f"  B_total:             {params.diffusion.B_total} μM")
    print(f"  K_D_buffer:          {params.diffusion.K_D_buffer} μM")
    print(f"  Buffering ratio:     {1 + params.diffusion.B_total/params.diffusion.K_D_buffer:.1f}x")

    print(f"\nEnzymes:")
    print(f"  AC V_basal:          {params.enzyme.V_basal} μM/s")
    print(f"  AC V_max:            {params.enzyme.V_max_AC} μM/s")
    print(f"  PDE k_cat:           {params.enzyme.k_cat} s⁻¹")
    print(f"  PDE K_m:             {params.enzyme.K_m} μM")
    print(f"  PDE V_max:           {params.effective_V_max_PDE():.1f} μM/s")

    # Absorptive action (Lohse et al.)
    R_PDE = 2.5e-3  # μm
    D_eff = params.diffusion.D_eff_low
    eta = params.enzyme.k_cat / (4 * np.pi * R_PDE * D_eff * params.enzyme.K_m)
    print(f"\n  Absorptive action η: {eta:.4f} (using D_eff)")
    eta_free = params.enzyme.k_cat / (4 * np.pi * R_PDE * params.diffusion.D_free * params.enzyme.K_m)
    print(f"  Absorptive action η: {eta_free:.6f} (using D_free)")

    # Geometric coupling rate
    k_coupling = (params.diffusion.D_free * params.geometry.axon_cross_section /
                  (params.geometry.bouton_volume * params.geometry.inter_bouton_distance))
    print(f"\n  Geometric coupling:  {k_coupling:.3f} s⁻¹ (bouton-to-bouton via D_free)")
    k_coupling_eff = (D_eff * params.geometry.axon_cross_section /
                      (params.geometry.bouton_volume * params.geometry.inter_bouton_distance))
    print(f"  Geometric coupling:  {k_coupling_eff:.4f} s⁻¹ (via D_eff)")

    print(f"\nStimulation:")
    print(f"  Pairings:            {params.stimulation.n_pairings}")
    print(f"  ITI:                 {params.stimulation.inter_trial_interval} s")
    print(f"  Aversive comps:      {[f'γ{c+1}' for c in params.stimulation.aversive_compartments]}")


if __name__ == '__main__':
    # Print model summary
    params = default_params()
    print_model_summary(params)

    # Run experiments
    result_wt, result_dunce = run_wt_vs_dunce()
    run_kcat_sensitivity()
    run_buffering_sensitivity()
    run_pde_dose_response()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Output files in: {OUTPUT_DIR}")
    print("=" * 60)
