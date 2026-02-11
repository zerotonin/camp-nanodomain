"""
Spatial geometry for 1D compartmental model of KC axon.

Builds a 1D grid alternating between boutons (well-mixed, single bin)
and axon segments (multiple bins). Tracks cross-sectional area A(x),
element type, and compartment assignment for each grid point.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .parameters import GeometryParams


# Element type identifiers
BOUTON = 0
AXON = 1


@dataclass
class SpatialGrid:
    """1D spatial grid with geometry information.

    Attributes:
        x: Position of each grid point (μm)
        dx: Spacing between grid points (μm) — variable
        A: Cross-sectional area at each grid point (μm²)
        element_type: BOUTON or AXON for each point
        compartment_id: Which γ-compartment (0–4) or -1 for axon
        bouton_indices: List of arrays, one per compartment, giving grid indices
        axon_segment_indices: List of arrays for each inter-bouton axon segment
        n_total: Total number of grid points
    """
    x: np.ndarray
    dx: np.ndarray
    A: np.ndarray
    element_type: np.ndarray
    compartment_id: np.ndarray
    bouton_indices: List[np.ndarray]
    axon_segment_indices: List[np.ndarray]
    n_total: int

    def bouton_center_positions(self) -> np.ndarray:
        """Return x-positions of bouton centers."""
        return np.array([self.x[idx].mean() for idx in self.bouton_indices])

    def get_bouton_mean_index(self, compartment: int) -> int:
        """Return the central grid index for a given bouton."""
        idx = self.bouton_indices[compartment]
        return idx[len(idx) // 2]


def build_grid(geom: GeometryParams) -> SpatialGrid:
    """Construct the 1D spatial grid.

    Layout: [Bouton_0] [Axon_01] [Bouton_1] [Axon_12] ... [Bouton_N-1]

    Boutons are represented as a single well-mixed bin.
    Axon segments are subdivided into axon_bins_per_segment bins.
    """
    N = geom.n_compartments
    n_axon_bins = geom.axon_bins_per_segment
    axon_length = geom.axon_segment_length  # length of each axon segment
    axon_dx = geom.axon_dx
    bouton_d = geom.bouton_diameter

    # Build arrays incrementally
    positions = []
    dx_arr = []
    areas = []
    types = []
    comp_ids = []
    bouton_indices = []
    axon_indices = []

    current_x = 0.0
    idx = 0

    for i in range(N):
        # --- Bouton i ---
        start_idx = idx
        # Single well-mixed bin for bouton
        positions.append(current_x + bouton_d / 2.0)
        dx_arr.append(bouton_d)  # effective "width" of bouton bin
        areas.append(geom.bouton_cross_section)
        types.append(BOUTON)
        comp_ids.append(i)
        idx += 1

        bouton_indices.append(np.array([start_idx]))

        current_x += bouton_d

        # --- Axon segment between bouton i and i+1 ---
        if i < N - 1:
            axon_start_idx = idx
            for j in range(n_axon_bins):
                positions.append(current_x + (j + 0.5) * axon_dx)
                dx_arr.append(axon_dx)
                areas.append(geom.axon_cross_section)
                types.append(AXON)
                comp_ids.append(-1)  # axon, no compartment
                idx += 1
            axon_indices.append(np.arange(axon_start_idx, idx))
            current_x += axon_length

    grid = SpatialGrid(
        x=np.array(positions),
        dx=np.array(dx_arr),
        A=np.array(areas),
        element_type=np.array(types, dtype=int),
        compartment_id=np.array(comp_ids, dtype=int),
        bouton_indices=bouton_indices,
        axon_segment_indices=axon_indices,
        n_total=idx,
    )

    return grid


def compute_coupling_coefficients(grid: SpatialGrid, D: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute diffusive coupling between adjacent grid points.

    Returns:
        alpha_plus: Coupling coefficient to right neighbor (s⁻¹)
        alpha_minus: Coupling coefficient to left neighbor (s⁻¹)

    The diffusion term for bin i is:
        d(c_i)/dt = alpha_plus[i]*(c_{i+1} - c_i) - alpha_minus[i]*(c_i - c_{i-1})

    Accounting for variable cross-sections (bouton↔axon junctions):
        flux = D * A_interface * (c_j - c_i) / distance
        dc_i/dt += flux / V_i
    where V_i = A_i * dx_i is the volume of bin i.
    """
    n = grid.n_total
    alpha_plus = np.zeros(n)
    alpha_minus = np.zeros(n)

    for i in range(n - 1):
        # Interface area: harmonic mean of adjacent cross-sections
        A_interface = 2.0 * grid.A[i] * grid.A[i + 1] / (grid.A[i] + grid.A[i + 1])
        # Distance between bin centers
        dist = 0.5 * (grid.dx[i] + grid.dx[i + 1])
        # Flux coefficient
        flux_coeff = D * A_interface / dist

        # Volume of each bin
        V_i = grid.A[i] * grid.dx[i]
        V_ip1 = grid.A[i + 1] * grid.dx[i + 1]

        # Coupling rates
        alpha_plus[i] = flux_coeff / V_i
        alpha_minus[i + 1] = flux_coeff / V_ip1

    return alpha_plus, alpha_minus
