"""
Stimulation protocols for the cAMP nanodomain model.

Generates time-varying Ca²⁺ influx and odor signals for
aversive conditioning training protocols.
"""

import numpy as np
from .parameters import ModelParams
from .geometry import SpatialGrid


class StimulationProtocol:
    """Manages odor and shock (Ca²⁺) stimulation across trials.

    The training protocol consists of n_pairings of:
      - Odor presentation (activates KC → AC in all boutons)
      - Shock (activates DANs → Ca²⁺ in γ1–γ2 boutons)

    The shock is delivered during the last portion of odor (coincidence).
    """

    def __init__(self, params: ModelParams, grid: SpatialGrid):
        self.params = params
        self.grid = grid
        self.stim = params.stimulation
        self.ca_params = params.calcium

        # Precompute trial onset times
        self.trial_onsets = np.array([
            i * self.stim.inter_trial_interval
            for i in range(self.stim.n_pairings)
        ])

        # Total simulation time: last trial + buffer for decay
        self.total_time = (
            self.trial_onsets[-1]
            + self.stim.odor_duration
            + 20.0  # extra time to see decay/spreading
        )

    def odor_signal(self, t: float) -> float:
        """Return odor activation at time t (0 or 1).

        Odor is on during [trial_onset, trial_onset + odor_duration]
        for each trial.
        """
        for onset in self.trial_onsets:
            if onset <= t < onset + self.stim.odor_duration:
                return 1.0
        return 0.0

    def shock_signal(self, t: float) -> float:
        """Return shock (DAN activation) signal at time t (0 or 1).

        Shock occurs during [trial_onset + shock_onset, trial_onset + shock_onset + shock_duration].
        """
        for onset in self.trial_onsets:
            shock_start = onset + self.stim.shock_onset
            shock_end = shock_start + self.stim.shock_duration
            if shock_start <= t < shock_end:
                return 1.0
        return 0.0

    def calcium_influx(self, t: float, grid_index: int) -> float:
        """Return Ca²⁺ influx rate (μM/s) at grid point i and time t.

        Ca²⁺ influx occurs only in aversive compartment boutons during shock.
        """
        # Only boutons in aversive compartments get Ca²⁺
        comp = self.grid.compartment_id[grid_index]
        if comp < 0 or comp not in self.stim.aversive_compartments:
            return 0.0

        if self.shock_signal(t) > 0:
            return self.ca_params.Ca_amplitude / self.ca_params.Ca_pulse_duration
        return 0.0

    def ac_production_rate(self, t: float, grid_index: int,
                           ca_conc: float) -> float:
        """Return cAMP production rate (μM/s) at grid point i.

        J_AC = V_basal + V_max_AC * f_Ca * f_odor

        AC is only in boutons. Odor activates AC in all boutons.
        Ca²⁺ enhances AC in boutons receiving DAN input.
        """
        comp = self.grid.compartment_id[grid_index]

        # No AC in axon segments
        if comp < 0:
            return 0.0

        enz = self.params.enzyme

        # Basal production always present in boutons
        j_ac = enz.V_basal

        # Ca²⁺ activation (Hill function)
        f_ca = ca_conc**enz.n_Hill_AC / (
            enz.K_Ca**enz.n_Hill_AC + ca_conc**enz.n_Hill_AC
        )

        # Odor-dependent enhancement
        f_odor = self.odor_signal(t) if comp in self.stim.odor_compartments else 0.0

        j_ac += enz.V_max_AC * f_ca * f_odor

        return j_ac
