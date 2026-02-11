"""
Biophysical parameters for cAMP nanodomain model.

All concentrations in μM, distances in μm, times in seconds.
Sources: Lohse et al. 2017 (PLOS ONE), Bock et al. 2020 (Cell),
         Bender & Beavo 2006, Walker-Gray et al. 2017.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class GeometryParams:
    """Spatial geometry of the KC axon."""
    n_compartments: int = 5                # γ1 through γ5
    bouton_diameter: float = 1.5           # μm
    axon_diameter: float = 0.3             # μm
    inter_bouton_distance: float = 5.0     # μm (center-to-center)
    axon_bins_per_segment: int = 20        # spatial resolution in axon

    @property
    def bouton_radius(self) -> float:
        return self.bouton_diameter / 2.0

    @property
    def bouton_volume(self) -> float:
        """Bouton volume (μm³), modeled as sphere."""
        return (4.0 / 3.0) * np.pi * self.bouton_radius**3

    @property
    def bouton_cross_section(self) -> float:
        """Cross-sectional area of bouton at equator (μm²)."""
        return np.pi * self.bouton_radius**2

    @property
    def axon_cross_section(self) -> float:
        """Cross-sectional area of axon (μm²)."""
        return np.pi * (self.axon_diameter / 2.0)**2

    @property
    def axon_segment_length(self) -> float:
        """Length of axon between boutons (μm), excluding bouton radii."""
        return self.inter_bouton_distance - self.bouton_diameter

    @property
    def axon_dx(self) -> float:
        """Spatial step in axon segments (μm)."""
        return self.axon_segment_length / self.axon_bins_per_segment


@dataclass
class DiffusionParams:
    """cAMP diffusion and buffering parameters."""
    D_free: float = 130.0        # μm²/s — free cAMP diffusion (Bock 2020)
    B_total: float = 20.0        # μM — total buffer concentration (Bock 2020, conservative)
    K_D_buffer: float = 2.0      # μM — buffer dissociation constant (PKA-RIα)
    k_on_buffer: float = 10.0    # μM⁻¹s⁻¹ — association rate (≈10⁷ M⁻¹s⁻¹)

    @property
    def k_off_buffer(self) -> float:
        """Dissociation rate (s⁻¹)."""
        return self.k_on_buffer * self.K_D_buffer

    def D_eff(self, c_free: float = 0.0) -> float:
        """Effective diffusion coefficient accounting for buffering (μm²/s).

        At low cAMP: D_eff ≈ D_free / (1 + B_total/K_D_buffer)
        """
        denom = 1.0 + self.B_total * self.K_D_buffer / (self.K_D_buffer + c_free)**2
        return self.D_free / denom

    @property
    def D_eff_low(self) -> float:
        """Effective diffusion at low [cAMP] (μm²/s)."""
        return self.D_free / (1.0 + self.B_total / self.K_D_buffer)


@dataclass
class EnzymeParams:
    """Adenylyl cyclase and PDE (dunce) parameters."""
    # Adenylyl cyclase
    V_basal: float = 0.5         # μM/s — basal cAMP production
    V_max_AC: float = 10.0       # μM/s — maximal Ca²⁺-stimulated production
    K_Ca: float = 0.5            # μM — Ca²⁺ half-activation of AC
    n_Hill_AC: float = 2.0       # Hill coefficient for Ca²⁺/CaM

    # Dunce / PDE4
    k_cat: float = 5.0           # s⁻¹ — catalytic rate (literature: 5, in-cell: ~160)
    K_m: float = 2.4             # μM — Michaelis constant
    PDE_concentration: float = 1.0  # μM — [PDE] in boutons (adjustable)

    @property
    def V_max_PDE(self) -> float:
        """Maximal PDE degradation rate in boutons (μM/s)."""
        return self.PDE_concentration * self.k_cat


@dataclass
class CalciumParams:
    """Intracellular calcium dynamics."""
    Ca_rest: float = 0.05        # μM — resting [Ca²⁺]
    Ca_amplitude: float = 2.0    # μM — peak [Ca²⁺] per shock pulse
    tau_Ca: float = 1.0          # s — Ca²⁺ clearance time constant
    Ca_pulse_duration: float = 0.1  # s — duration of each Ca²⁺ influx event


@dataclass
class StimulationParams:
    """Training protocol parameters."""
    odor_duration: float = 5.0        # s
    shock_onset: float = 4.0          # s — shock starts during last second of odor
    shock_duration: float = 1.0       # s
    n_pairings: int = 6               # number of odor-shock pairings
    inter_trial_interval: float = 30.0  # s between trial onsets
    # Which compartments receive DAN Ca²⁺ (0-indexed: γ1=0, γ5=4)
    aversive_compartments: List[int] = field(default_factory=lambda: [0, 1])
    # Odor activates AC in all compartments (KC is activated by odor)
    odor_compartments: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclass
class ModelParams:
    """Complete parameter set for the model."""
    geometry: GeometryParams = field(default_factory=GeometryParams)
    diffusion: DiffusionParams = field(default_factory=DiffusionParams)
    enzyme: EnzymeParams = field(default_factory=EnzymeParams)
    calcium: CalciumParams = field(default_factory=CalciumParams)
    stimulation: StimulationParams = field(default_factory=StimulationParams)

    # Model variant
    is_dunce_mutant: bool = False    # If True, V_max_PDE = 0 everywhere
    pde_inhibition: float = 0.0      # Fraction inhibited (0 = full activity, 1 = complete block)

    def effective_V_max_PDE(self) -> float:
        """PDE activity accounting for genotype and inhibition."""
        if self.is_dunce_mutant:
            return 0.0
        return self.enzyme.V_max_PDE * (1.0 - self.pde_inhibition)


def default_params() -> ModelParams:
    """Return default wild-type parameters."""
    return ModelParams()


def dunce_mutant_params() -> ModelParams:
    """Return dunce⁻ mutant parameters."""
    return ModelParams(is_dunce_mutant=True)


def high_kcat_params() -> ModelParams:
    """Return parameters with in-cell k_cat estimate (~160 s⁻¹)."""
    p = ModelParams()
    p.enzyme.k_cat = 160.0
    return p
