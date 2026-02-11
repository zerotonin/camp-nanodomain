"""
cAMP Nanodomain Model

A 1D reaction-diffusion model of cAMP compartmentalization in
Drosophila Kenyon cell gamma lobes, based on the Lohse lab
framework for PDE-mediated nanodomains.
"""

from .parameters import (
    ModelParams, GeometryParams, DiffusionParams, EnzymeParams,
    CalciumParams, StimulationParams,
    default_params, dunce_mutant_params, high_kcat_params,
)
from .geometry import SpatialGrid, build_grid, BOUTON, AXON
from .stimulation import StimulationProtocol
from .dynamics import Dynamics
from .simulation import run_simulation, SimulationResult
from .visualization import (
    plot_kymograph, plot_bouton_timecourses, plot_spatial_profiles,
    plot_comparison, plot_spatial_comparison, plot_cumulative_spreading,
)

__all__ = [
    'ModelParams', 'GeometryParams', 'DiffusionParams', 'EnzymeParams',
    'CalciumParams', 'StimulationParams',
    'default_params', 'dunce_mutant_params', 'high_kcat_params',
    'SpatialGrid', 'build_grid', 'BOUTON', 'AXON',
    'StimulationProtocol', 'Dynamics',
    'run_simulation', 'SimulationResult',
    'plot_kymograph', 'plot_bouton_timecourses', 'plot_spatial_profiles',
    'plot_comparison', 'plot_spatial_comparison', 'plot_cumulative_spreading',
]
