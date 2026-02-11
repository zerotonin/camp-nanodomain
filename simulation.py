"""
Time integration for the cAMP nanodomain model.

Uses scipy.integrate.solve_ivp with stiff solvers (BDF/Radau)
to handle fast buffering kinetics.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional

from .parameters import ModelParams
from .geometry import SpatialGrid, build_grid
from .stimulation import StimulationProtocol
from .dynamics import Dynamics


@dataclass
class SimulationResult:
    """Container for simulation output.

    Attributes:
        t: Time points (s)
        c_free: Free cAMP [n_time, n_grid] (μM)
        c_bound: Bound cAMP [n_time, n_grid] (μM)
        ca: Calcium [n_time, n_grid] (μM)
        c_total: Total cAMP (free + bound) [n_time, n_grid] (μM)
        grid: Spatial grid
        params: Model parameters
        bouton_c_free: Free cAMP averaged over each bouton [n_time, n_compartments]
        bouton_c_total: Total cAMP averaged over each bouton [n_time, n_compartments]
    """
    t: np.ndarray
    c_free: np.ndarray
    c_bound: np.ndarray
    ca: np.ndarray
    c_total: np.ndarray
    grid: SpatialGrid
    params: ModelParams
    bouton_c_free: np.ndarray
    bouton_c_total: np.ndarray


def run_simulation(params: ModelParams,
                   method: str = 'BDF',
                   max_step: float = 0.1,
                   rtol: float = 1e-6,
                   atol: float = 1e-9,
                   dense_output_dt: Optional[float] = 0.1,
                   ) -> SimulationResult:
    """Run the full simulation.

    Args:
        params: Model parameters (WT or dunce⁻)
        method: Integration method ('BDF', 'Radau', 'RK45')
        max_step: Maximum time step (s)
        rtol: Relative tolerance
        atol: Absolute tolerance
        dense_output_dt: If set, evaluate solution at uniform time steps

    Returns:
        SimulationResult with all state variables over time
    """
    # Build spatial grid
    grid = build_grid(params.geometry)

    # Create stimulation protocol
    protocol = StimulationProtocol(params, grid)

    # Create dynamics
    dyn = Dynamics(params, grid, protocol)

    # Initial conditions
    y0 = dyn.initial_conditions()

    # Time span
    t_span = (0.0, protocol.total_time)

    # Evaluation time points
    if dense_output_dt is not None:
        t_eval = np.arange(0.0, protocol.total_time, dense_output_dt)
    else:
        t_eval = None

    print(f"Running simulation: {params.is_dunce_mutant and 'dunce⁻' or 'WT'}")
    print(f"  Grid points: {grid.n_total}")
    print(f"  State dimension: {dyn.state_size}")
    print(f"  Total time: {protocol.total_time:.1f} s")
    print(f"  Pairings: {params.stimulation.n_pairings}")
    print(f"  Method: {method}")

    # Integrate
    sol = solve_ivp(
        dyn.rhs,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    print(f"  Completed: {sol.t.shape[0]} time points, status={sol.status}")

    # Unpack results
    N = grid.n_total
    t = sol.t
    c_free = sol.y[:N, :].T        # [n_time, n_grid]
    c_bound = sol.y[N:2*N, :].T
    ca = sol.y[2*N:3*N, :].T
    c_total = c_free + c_bound

    # Compute bouton-averaged concentrations
    n_comp = params.geometry.n_compartments
    bouton_c_free = np.zeros((len(t), n_comp))
    bouton_c_total = np.zeros((len(t), n_comp))
    for k in range(n_comp):
        idx = grid.bouton_indices[k]
        bouton_c_free[:, k] = c_free[:, idx].mean(axis=1)
        bouton_c_total[:, k] = c_total[:, idx].mean(axis=1)

    return SimulationResult(
        t=t,
        c_free=c_free,
        c_bound=c_bound,
        ca=ca,
        c_total=c_total,
        grid=grid,
        params=params,
        bouton_c_free=bouton_c_free,
        bouton_c_total=bouton_c_total,
    )
