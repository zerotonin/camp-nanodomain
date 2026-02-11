#!/bin/bash
#SBATCH --job-name=camp_optim_v2
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=03:30:00
#SBATCH --mem=16GB
#SBATCH --output=camp_optim_%j.log
#SBATCH --error=camp_optim_%j.err

# ============================================================
# cAMP Nanodomain Model — GPU Parameter Optimisation v2
# ============================================================
#
# Setup (run ONCE before first submission):
#
#   conda create -n camp-nanodomain python=3.11 numpy scipy pandas matplotlib tqdm -c conda-forge -y
#   conda activate camp-nanodomain
#   pip install "jax[cuda12]==0.4.30" "diffrax>=0.5.0,<0.7" "equinox>=0.11,<0.12" "optax>=0.2.2"
#   python -c "import jax; print(jax.devices())"   # should show [CudaDevice(id=0)]
#
# Submit:
#   sbatch slurm_camp_optimise.sh
#
# Quick test (50 generations):
#   sbatch --time=00:30:00 --export=ALL,QUICK=1 slurm_camp_optimise.sh
#
# ============================================================

# --- Paths (edit these) ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/cAMP_model"
CODE_DIR="/home/geuba03p/PyProjects/camp-nanodomain"
DATA_FILE="${CODE_DIR}/all_camp_long.csv"
SCRIPT="${CODE_DIR}/optimise_model_gpu_v2.py"
OUTDIR="${PROJECT_DIR}/results_v2_${SLURM_JOB_ID}"

# --- Activate environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camp-nanodomain

# --- Diagnostics ---
echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:       $(which python)"
echo "JAX devices:  $(python -c 'import jax; print(jax.devices())' 2>&1)"
echo "Output dir:   ${OUTDIR}"
echo "============================================"

mkdir -p "${OUTDIR}"

# --- Build command ---
CMD="python ${SCRIPT} --data ${DATA_FILE} --outdir ${OUTDIR}"
CMD="${CMD} --maxiter 300 --popsize 25 --seed ${SLURM_JOB_ID:-42}"

# Quick mode (pass QUICK=1 via --export)
if [ "${QUICK}" = "1" ]; then
    CMD="${CMD} --quick"
    echo ">>> QUICK MODE (50 generations) <<<"
fi

echo "Running: ${CMD}"
echo "============================================"

# --- Run ---
time ${CMD}

echo ""
echo "============================================"
echo "Done. Results in: ${OUTDIR}"
echo "============================================"
