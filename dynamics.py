"""
ODE dynamics for cAMP reaction-diffusion model.

State vector layout for N grid points:
    y[0:N]       = c_free  (free cAMP, μM)
    y[N:2N]      = c_bound (buffer-bound cAMP, μM)
    y[2N:3N]     = Ca²⁺    (intracellular calcium, μM)

Total state dimension: 3 * N_grid_points
"""

import numpy as np
from .parameters import ModelParams
from .geometry import SpatialGrid, BOUTON, compute_coupling_coefficients
from .stimulation import StimulationProtocol


class Dynamics:
    """Computes the ODE right-hand side for the full reaction-diffusion system."""

    def __init__(self, params: ModelParams, grid: SpatialGrid,
                 protocol: StimulationProtocol):
        self.params = params
        self.grid = grid
        self.protocol = protocol
        self.N = grid.n_total

        # Precompute coupling coefficients for diffusion
        self.alpha_plus, self.alpha_minus = compute_coupling_coefficients(
            grid, params.diffusion.D_free
        )

        # Precompute which bins are boutons (for PDE activity)
        self.is_bouton = (grid.element_type == BOUTON)

        # PDE activity per bin (μM/s Vmax) — only in boutons, 0 in axon
        self.V_max_PDE = np.zeros(self.N)
        V_max = params.effective_V_max_PDE()
        self.V_max_PDE[self.is_bouton] = V_max

    @property
    def state_size(self) -> int:
        return 3 * self.N

    def pack_state(self, c_free: np.ndarray, c_bound: np.ndarray,
                   ca: np.ndarray) -> np.ndarray:
        """Pack state variables into a single vector."""
        return np.concatenate([c_free, c_bound, ca])

    def unpack_state(self, y: np.ndarray):
        """Unpack state vector into (c_free, c_bound, Ca²⁺)."""
        N = self.N
        return y[:N].copy(), y[N:2*N].copy(), y[2*N:3*N].copy()

    def initial_conditions(self) -> np.ndarray:
        """Set initial state: resting conditions."""
        N = self.N
        c_free = np.zeros(N)
        c_bound = np.zeros(N)
        ca = np.full(N, self.params.calcium.Ca_rest)
        return self.pack_state(c_free, c_bound, ca)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute dy/dt for the full system.

        Equations:
            dc_free/dt  = diffusion + J_AC - J_PDE - J_bind + J_unbind
            dc_bound/dt = J_bind - J_unbind
            d[Ca²⁺]/dt = J_Ca_influx - ([Ca²⁺] - Ca_rest)/tau_Ca
        """
        N = self.N
        c_free = y[:N]
        c_bound = y[N:2*N]
        ca = y[2*N:3*N]

        # Initialize derivatives
        dc_free = np.zeros(N)
        dc_bound = np.zeros(N)
        dca = np.zeros(N)

        # --- 1. DIFFUSION of free cAMP ---
        # Interior points: dc/dt += alpha_plus*(c_{i+1} - c_i) - alpha_minus*(c_i - c_{i-1})
        # Using vectorized operations for the bulk
        # Right flux: alpha_plus[i] * (c[i+1] - c[i])
        dc_free[:-1] += self.alpha_plus[:-1] * (c_free[1:] - c_free[:-1])
        # Left flux: alpha_minus[i] * (c[i-1] - c[i])  →  same sign convention
        dc_free[1:] += self.alpha_minus[1:] * (c_free[:-1] - c_free[1:])
        # Boundary conditions: no-flux at ends (nothing to add)

        # --- 2. cAMP PRODUCTION (AC) ---
        # Compute per-bin: only boutons have AC
        for i in range(N):
            if self.is_bouton[i]:
                j_ac = self.protocol.ac_production_rate(t, i, ca[i])
                dc_free[i] += j_ac

        # --- 3. cAMP DEGRADATION (dunce/PDE4) ---
        # Michaelis-Menten: J_PDE = V_max_PDE * c_free / (K_m + c_free)
        K_m = self.params.enzyme.K_m
        # Avoid division issues with negative concentrations
        c_pos = np.maximum(c_free, 0.0)
        j_pde = self.V_max_PDE * c_pos / (K_m + c_pos)
        dc_free -= j_pde

        # --- 4. BUFFERING (binding/unbinding) ---
        diff = self.params.diffusion
        B_free = np.maximum(diff.B_total - c_bound, 0.0)
        j_bind = diff.k_on_buffer * c_pos * B_free
        j_unbind = diff.k_off_buffer * np.maximum(c_bound, 0.0)

        dc_free += -j_bind + j_unbind
        dc_bound += j_bind - j_unbind

        # --- 5. CALCIUM DYNAMICS ---
        tau_ca = self.params.calcium.tau_Ca
        ca_rest = self.params.calcium.Ca_rest

        # Decay toward resting
        dca -= (ca - ca_rest) / tau_ca

        # Ca²⁺ influx during shock (only aversive boutons)
        for i in range(N):
            j_ca = self.protocol.calcium_influx(t, i)
            dca[i] += j_ca

        return np.concatenate([dc_free, dc_bound, dca])
