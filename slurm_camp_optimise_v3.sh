#!/bin/bash
#SBATCH --job-name=camp_optim_v3
#SBATCH --account=geuba03p
#SBATCH --partition=aoraki_gpu_H100,aoraki_gpu,aoraki_gpu_L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=03:30:00
#SBATCH --mem=16GB
#SBATCH --output=camp_optim_v3_%j.log
#SBATCH --error=camp_optim_v3_%j.err

# ============================================================
# cAMP Nanodomain Model — GPU Parameter Optimisation v3
# ============================================================
#
# Two-term AC model: Ca²⁺-only + Coincidence (Ca²⁺+DA)
# DA mask: [1.0, 1.0, 0.5, 0.0, 0.0] for γ1-γ5
# Ca²⁺ driven by odour; τ_sensor bounds tightened to [0.5, 10] s
# 11 fitted parameters (was 10 in v2)
#
# Submit:
#   sbatch slurm_camp_optimise_v3.sh
#
# Quick test (50 generations):
#   sbatch --time=00:30:00 --export=ALL,QUICK=1 slurm_camp_optimise_v3.sh
#
# ============================================================

# --- Paths (edit these) ---
PROJECT_DIR="/projects/sciences/zoology/geurten_lab/cAMP_model"
CODE_DIR="/home/geuba03p/PyProjects/camp-nanodomain"
DATA_FILE="${CODE_DIR}/all_camp_long.csv"
SCRIPT="${CODE_DIR}/optimise_model_gpu_v3.py"
OUTDIR="${PROJECT_DIR}/results_v3_${SLURM_JOB_ID}"

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
echo "Model:        V3: Two-term AC with DA mask and tighter τ_sensor bounds"
echo "============================================"

# --- Verify files exist ---
if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: Script not found: ${SCRIPT}"
    echo "  Make sure optimise_model_gpu_v3.py is in ${CODE_DIR}/"
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

# Quick mode (pass QUICK=1 via --export)
if [ "${QUICK}" = "1" ]; then
    CMD="${CMD} --quick"
    echo ">>> QUICK MODE (50 generations) <<<"
fi

echo "Running: ${CMD}"
echo "============================================"

# --- Run main optimisation ---
time ${CMD}

RC=$?
if [ $RC -ne 0 ]; then
    echo "ERROR: Main optimisation failed with exit code ${RC}"
    exit $RC
fi

# --- Multi-seed runs for robustness ---
for SEED in 123 456 789; do
    echo ""
    echo "--- Seed ${SEED} ---"
    SEED_DIR="${OUTDIR}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"
    time python ${SCRIPT} \
        --data ${DATA_FILE} \
        --outdir "${SEED_DIR}" \
        --maxiter 300 \
        --popsize 25 \
        --seed ${SEED}
done

echo ""
echo "============================================"
echo "Done. Results in: ${OUTDIR}"
echo "============================================"