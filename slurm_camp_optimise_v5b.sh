#!/bin/bash
#SBATCH --job-name=camp_v5b_wide
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=16GB
#SBATCH --output=camp_optim_v5b_%j.log
#SBATCH --error=camp_optim_v5b_%j.err

# ============================================================
# cAMP Model v5b — WIDE BOUNDS EXPLORATION
# ============================================================
# All bounds opened wide. Let the optimizer tell us what the
# data demands, then negotiate biology back in.
# 12 fitted parameters, 400 generations, popsize 30.
#
# Submit:       sbatch slurm_camp_optimise_v5b.sh
# Quick test:   sbatch --time=01:00:00 --export=ALL,QUICK=1 slurm_camp_optimise_v5b.sh
# ============================================================

export PYTHONUNBUFFERED=1

# --- Paths ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/cAMP_model"
CODE_DIR="/home/geuba03p/PyProjects/camp-nanodomain"
DATA_FILE="${CODE_DIR}/all_camp_long.csv"
SCRIPT="${CODE_DIR}/optimise_model_gpu_v5b.py"
OUTDIR="${PROJECT_DIR}/results_v5b_${SLURM_JOB_ID}"

# --- Activate environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate camp-nanodomain

# --- Diagnostics ---
echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:       $(which python)"
echo "JAX devices:  $(python -c 'import jax; print(jax.devices())' 2>&1)"
echo "Script:       ${SCRIPT}"
echo "Output dir:   ${OUTDIR}"
echo "Model:        V5b: WIDE BOUNDS — all params unconstrained"
echo "============================================"

# --- Verify ---
if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: Script not found: ${SCRIPT}"
    exit 1
fi
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data not found: ${DATA_FILE}"
    exit 1
fi

mkdir -p "${OUTDIR}"

# --- Run ---
CMD="python ${SCRIPT} --data ${DATA_FILE} --outdir ${OUTDIR}"
CMD="${CMD} --maxiter 400 --popsize 30 --seed ${SLURM_JOB_ID:-42}"

if [ "${QUICK}" = "1" ]; then
    CMD="${CMD} --quick"
    echo ">>> QUICK MODE (50 generations) <<<"
fi

echo "Running: ${CMD}"
echo "============================================"

time ${CMD}

RC=$?
if [ $RC -ne 0 ]; then
    echo "ERROR: Failed with exit code ${RC}"
    exit $RC
fi

echo ""
echo "============================================"
echo "Done. Results in: ${OUTDIR}"
echo "============================================"
