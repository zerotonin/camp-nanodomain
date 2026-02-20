#!/bin/bash
#SBATCH --job-name=camp_optim_v5
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=16GB
#SBATCH --output=camp_optim_v5_%j.log
#SBATCH --error=camp_optim_v5_%j.err

# Force unbuffered Python output so progress appears in the log in real time
export PYTHONUNBUFFERED=1

# ============================================================
# cAMP Nanodomain Model v5 — RBA Ca²⁺ + Free K_Ca
# ============================================================
#
# Key change from v4: K_Ca is now a FREE parameter [0.03, 0.5] μM
# This fixes the root cause of v3/v4 failure where the Hill function
# was in its dead zone because fixed K_Ca >> free [Ca²⁺] after RBA.
# 12 fitted parameters (was 11).
#
# Submit:       sbatch slurm_camp_optimise_v5.sh
# Quick test:   sbatch --time=00:45:00 --export=ALL,QUICK=1 slurm_camp_optimise_v5.sh
#
# ============================================================

# --- Paths ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/cAMP_model"
CODE_DIR="/home/geuba03p/PyProjects/camp-nanodomain"
DATA_FILE="${CODE_DIR}/all_camp_long.csv"
SCRIPT="${CODE_DIR}/optimise_model_gpu_v5.py"
OUTDIR="${PROJECT_DIR}/results_v5_${SLURM_JOB_ID}"

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
echo "Script:       ${SCRIPT}"
echo "Data:         ${DATA_FILE}"
echo "Output dir:   ${OUTDIR}"
echo "Model:        V5: RBA Ca²⁺ + FREE K_Ca [0.03-0.5 μM] + Two-term AC + DA mask"
echo "============================================"

# --- Verify files exist ---
if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: Script not found: ${SCRIPT}"
    echo "  Copy optimise_model_gpu_v5.py to ${CODE_DIR}/"
    exit 1
fi
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data not found: ${DATA_FILE}"
    exit 1
fi

mkdir -p "${OUTDIR}"

# --- Build command ---
CMD="python ${SCRIPT} --data ${DATA_FILE} --outdir ${OUTDIR}"
CMD="${CMD} --maxiter 300 --popsize 25 --seed ${SLURM_JOB_ID:-42}"

if [ "${QUICK}" = "1" ]; then
    CMD="${CMD} --quick"
    echo ">>> QUICK MODE (50 generations) <<<"
fi

echo "Running: ${CMD}"
echo "============================================"

# --- Run ---
time ${CMD}

RC=$?
if [ $RC -ne 0 ]; then
    echo "ERROR: Optimisation failed with exit code ${RC}"
    echo "Check camp_optim_v5_${SLURM_JOB_ID}.err for details"
    exit $RC
fi

echo ""
echo "============================================"
echo "Done. Results in: ${OUTDIR}"
echo "============================================"
