"""
Visualization for the cAMP nanodomain model.

Produces:
1. Kymographs (cAMP vs position vs time)
2. Bouton time courses (cAMP in γ1-γ5 over training)
3. Spatial profiles at key time points
4. WT vs dunce⁻ side-by-side comparisons
5. Parameter sensitivity plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from typing import List, Optional, Tuple

from .simulation import SimulationResult


# Color scheme for compartments
COMP_COLORS = ['#e41a1c', '#ff7f00', '#999999', '#377eb8', '#4daf4a']
COMP_LABELS = ['γ1', 'γ2', 'γ3', 'γ4', 'γ5']


def plot_kymograph(result: SimulationResult, ax: Optional[plt.Axes] = None,
                   variable: str = 'c_free', vmax: Optional[float] = None,
                   title: Optional[str] = None,
                   cbar: bool = True) -> plt.Axes:
    """Plot kymograph: concentration (color) vs position (x) vs time (y).

    Args:
        result: Simulation result
        ax: Axes to plot on (creates new if None)
        variable: 'c_free', 'c_total', or 'ca'
        vmax: Color scale max
        title: Plot title
        cbar: Show colorbar
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    data_map = {
        'c_free': result.c_free,
        'c_total': result.c_total,
        'ca': result.ca,
    }
    data = data_map[variable]

    x = result.grid.x
    t = result.t

    if vmax is None:
        vmax = np.percentile(data, 99.5)

    im = ax.pcolormesh(x, t, data,
                       shading='auto',
                       cmap='hot',
                       norm=Normalize(vmin=0, vmax=vmax))

    # Mark bouton positions
    bouton_centers = result.grid.bouton_center_positions()
    for k, xb in enumerate(bouton_centers):
        ax.axvline(xb, color='cyan', alpha=0.3, linewidth=0.8, linestyle='--')
        ax.text(xb, t[-1] * 1.02, COMP_LABELS[k],
                ha='center', va='bottom', fontsize=9, color='cyan')

    # Mark trial times
    stim = result.params.stimulation
    for i in range(stim.n_pairings):
        t_start = i * stim.inter_trial_interval + stim.shock_onset
        ax.axhline(t_start, color='yellow', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('Position along axon (μm)')
    ax.set_ylabel('Time (s)')
    label_map = {'c_free': 'Free [cAMP]', 'c_total': 'Total [cAMP]', 'ca': '[Ca²⁺]'}
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{label_map[variable]} (μM)')

    if cbar:
        plt.colorbar(im, ax=ax, label='μM')

    return ax


def plot_bouton_timecourses(result: SimulationResult,
                            ax: Optional[plt.Axes] = None,
                            variable: str = 'c_free',
                            title: Optional[str] = None) -> plt.Axes:
    """Plot cAMP concentration in each bouton over time.

    Args:
        result: Simulation result
        ax: Axes to plot on
        variable: 'c_free' or 'c_total'
        title: Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    data = result.bouton_c_free if variable == 'c_free' else result.bouton_c_total
    t = result.t

    for k in range(result.params.geometry.n_compartments):
        ax.plot(t, data[:, k], color=COMP_COLORS[k],
                label=COMP_LABELS[k], linewidth=1.5)

    # Mark shock times
    stim = result.params.stimulation
    for i in range(stim.n_pairings):
        t_shock = i * stim.inter_trial_interval + stim.shock_onset
        ax.axvspan(t_shock, t_shock + stim.shock_duration,
                   color='red', alpha=0.08)

    ax.set_xlabel('Time (s)')
    label_map = {'c_free': 'Free [cAMP] (μM)', 'c_total': 'Total [cAMP] (μM)'}
    ax.set_ylabel(label_map.get(variable, 'Concentration (μM)'))
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)

    if title:
        ax.set_title(title)
    else:
        genotype = 'dunce⁻' if result.params.is_dunce_mutant else 'WT'
        ax.set_title(f'{genotype} — Bouton {label_map.get(variable, "cAMP")}')

    return ax


def plot_spatial_profiles(result: SimulationResult,
                          time_points: List[float],
                          ax: Optional[plt.Axes] = None,
                          variable: str = 'c_free',
                          title: Optional[str] = None) -> plt.Axes:
    """Plot cAMP concentration along axon at specific time points.

    Args:
        result: Simulation result
        time_points: List of times (s) to plot snapshots
        ax: Axes
        variable: 'c_free' or 'c_total'
        title: Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    data_map = {'c_free': result.c_free, 'c_total': result.c_total}
    data = data_map[variable]
    x = result.grid.x
    t = result.t

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(time_points)))

    for tp, color in zip(time_points, colors):
        idx = np.argmin(np.abs(t - tp))
        actual_t = t[idx]
        ax.plot(x, data[idx, :], color=color,
                label=f't = {actual_t:.1f} s', linewidth=1.5)

    # Mark bouton positions
    bouton_centers = result.grid.bouton_center_positions()
    for k, xb in enumerate(bouton_centers):
        ax.axvline(xb, color='gray', alpha=0.3, linewidth=0.8, linestyle='--')
        ax.text(xb, ax.get_ylim()[1] * 0.95, COMP_LABELS[k],
                ha='center', va='top', fontsize=9, color='gray')

    ax.set_xlabel('Position along axon (μm)')
    ax.set_ylabel(f'{variable} (μM)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(bottom=0)

    if title:
        ax.set_title(title)

    return ax


def plot_comparison(result_wt: SimulationResult,
                    result_dunce: SimulationResult,
                    save_path: Optional[str] = None,
                    ) -> plt.Figure:
    """Create comprehensive WT vs dunce⁻ comparison figure.

    4-panel layout:
        Top row: Kymographs (WT | dunce⁻)
        Bottom row: Bouton time courses (WT | dunce⁻)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Shared color scale
    vmax = max(
        np.percentile(result_wt.c_free, 99.5),
        np.percentile(result_dunce.c_free, 99.5),
    )

    # Top left: WT kymograph
    ax1 = fig.add_subplot(gs[0, 0])
    plot_kymograph(result_wt, ax=ax1, vmax=vmax, title='WT — Free [cAMP]')

    # Top right: dunce⁻ kymograph
    ax2 = fig.add_subplot(gs[0, 1])
    plot_kymograph(result_dunce, ax=ax2, vmax=vmax, title='dunce⁻ — Free [cAMP]')

    # Bottom left: WT bouton time courses
    ax3 = fig.add_subplot(gs[1, 0])
    plot_bouton_timecourses(result_wt, ax=ax3, title='WT — Bouton [cAMP]_free')

    # Bottom right: dunce⁻ bouton time courses
    ax4 = fig.add_subplot(gs[1, 1])
    plot_bouton_timecourses(result_dunce, ax=ax4, title='dunce⁻ — Bouton [cAMP]_free')

    fig.suptitle('cAMP Nanodomain Model: WT vs dunce⁻', fontsize=14, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved comparison figure to {save_path}")

    return fig


def plot_spatial_comparison(result_wt: SimulationResult,
                            result_dunce: SimulationResult,
                            time_points: List[float],
                            save_path: Optional[str] = None) -> plt.Figure:
    """Side-by-side spatial profiles at key time points."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    plot_spatial_profiles(result_wt, time_points, ax=axes[0],
                          title='WT — Spatial [cAMP]_free')
    plot_spatial_profiles(result_dunce, time_points, ax=axes[1],
                          title='dunce⁻ — Spatial [cAMP]_free')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved spatial comparison to {save_path}")

    return fig


def plot_cumulative_spreading(result: SimulationResult,
                              ax: Optional[plt.Axes] = None,
                              title: Optional[str] = None) -> plt.Axes:
    """Plot the ratio of cAMP in γ4-5 vs γ1-2 over time.

    This directly tests the spreading hypothesis: in dunce⁻, this
    ratio should increase over multiple pairings.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    t = result.t
    # cAMP in aversive compartments (γ1-2)
    c_aversive = result.bouton_c_free[:, 0:2].mean(axis=1)
    # cAMP in appetitive compartments (γ4-5)
    c_appetitive = result.bouton_c_free[:, 3:5].mean(axis=1)

    ax.plot(t, c_aversive, 'r-', label='γ1-2 (aversive)', linewidth=1.5)
    ax.plot(t, c_appetitive, 'b-', label='γ4-5 (appetitive)', linewidth=1.5)
    ax.plot(t, result.bouton_c_free[:, 2], color='gray',
            linestyle='--', label='γ3 (mixed)', linewidth=1.0)

    # Mark shock times
    stim = result.params.stimulation
    for i in range(stim.n_pairings):
        t_shock = i * stim.inter_trial_interval + stim.shock_onset
        ax.axvspan(t_shock, t_shock + stim.shock_duration,
                   color='red', alpha=0.08)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Free [cAMP] (μM)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)

    genotype = 'dunce⁻' if result.params.is_dunce_mutant else 'WT'
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{genotype} — Aversive vs Appetitive Compartment cAMP')

    return ax
