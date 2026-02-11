# cAMP Nanodomain Model — GPU-Accelerated Fitting

## Quick Setup

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate camp-nanodomain

# 2. Verify GPU access
python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)]

# 3. Run optimisation
python optimise_model_gpu.py --data all_camp_long.csv --outdir results_gpu/
```

## Usage

```bash
# Full run (~10-20 min on RTX 4070, vs ~2-4 hours on CPU)
python optimise_model_gpu.py --data all_camp_long.csv --outdir results_gpu/

# Quick sanity check (~2 min)
python optimise_model_gpu.py --data all_camp_long.csv --quick --outdir results_gpu/

# Custom stimulation timing
python optimise_model_gpu.py --data all_camp_long.csv --stim-onsets "0,30,60,90,120"

# Re-plot a saved fit
python optimise_model_gpu.py --plot-only results_gpu/best_params.json

# Force CPU if GPU causes issues
python optimise_model_gpu.py --data all_camp_long.csv --cpu

# CPU-only version (no JAX needed, uses scipy)
python optimise_model.py --data all_camp_long.csv --outdir results_cpu/
```

## What Gets GPU-Accelerated

| Component | CPU version | GPU version |
|-----------|------------|-------------|
| ODE integration | scipy BDF (sequential) | diffrax Kvaerno5 (JIT-compiled) |
| Population eval | Sequential loop | `jax.vmap` (all members in parallel) |
| Cost function | Python + numpy | JIT-compiled JAX |
| DE algorithm | scipy (CPU) | Custom JAX DE (runs on GPU) |

The GPU version runs the entire DE population (~180 forward solves = 2 genotypes × 90 members) in a single GPU kernel per generation.

## Troubleshooting

**JAX doesn't see GPU:**
```bash
# Check CUDA
nvidia-smi
# Reinstall JAX for your CUDA version
pip install --upgrade "jax[cuda12]"
```

**Out of memory (8GB VRAM):**
```bash
# Reduce population size
python optimise_model_gpu.py --popsize 10
```

**NaN in cost function:**
This usually means the ODE solver diverged for extreme parameter combinations.
The DE naturally selects away from these. If persistent, try:
```bash
# Increase data downsampling (fewer interpolation points)
python optimise_model_gpu.py --downsample-dt 4.0
```

## Output Files

- `best_params.json` — Fitted parameters + metadata
- `fit_result_gpu.png` — Model vs data comparison (both genotypes)
- `convergence_gpu.png` — Cost vs generation

## Modifying Fitted Parameters

Edit the `FIT_INDICES` and `FIT_BOUNDS` arrays near line 200 in `optimise_model_gpu.py`.
The indices map to `PARAM_NAMES`:

```
 0: D_free        1: B_total       2: K_D_buffer    3: k_on_buffer
 4: V_basal       5: V_max_AC      6: K_Ca          7: n_Hill_AC
 8: k_cat         9: K_m          10: PDE_conc      11: Ca_rest
12: Ca_amplitude  13: tau_Ca       14: Ca_pulse_dur  15: alpha
16: dff_offset
```

For example, to also fit D_free and K_m, add indices 0 and 9 to `FIT_INDICES`
and corresponding bounds to `FIT_BOUNDS`.
