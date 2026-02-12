#!/bin/bash
#SBATCH --job-name=camp_v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=camp_v3_%j.out
#SBATCH --error=camp_v3_%j.err

# ============================================================
# cAMP Nanodomain Model v3 — Two-Term AC (Ca²⁺-only + Coincidence)
# ============================================================
#
# Changes from v2:
#   - Two AC production terms: V_Ca (Ca²⁺-only, all compartments)
#     and V_coinc (coincidence Ca²⁺+DA, γ1-γ2 only)
#   - DA mask: [1.0, 1.0, 0.5, 0.0, 0.0] for γ1-γ5
#   - Ca²⁺ driven by odour (KC depolarisation), not shock
#   - Tighter τ_sensor bounds [0.5, 10] s
#   - 11 fitted parameters (was 10)
#
# Submit:  sbatch slurm_camp_optimise_v3.sh
# Monitor: tail -f camp_v3_<jobid>.out

module purge
module load python/3.11
module load cuda/12.2

source ~/venvs/jax-gpu/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_PLATFORM_NAME=gpu
export CUDA_VISIBLE_DEVICES=0

OUTDIR="results_v3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "=========================================="
echo "cAMP Model v3 — Two-Term AC"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Output:    $OUTDIR"
echo "=========================================="

# Main optimisation: 300 generations, popsize 25 for 11-D space
python optimise_model_gpu_v3.py \
    --data all_camp_long.csv \
    --outdir "$OUTDIR" \
    --maxiter 300 \
    --popsize 25 \
    --seed 42

# Multi-seed runs for robustness check
for SEED in 123 456 789; do
    echo ""
    echo "--- Seed $SEED ---"
    python optimise_model_gpu_v3.py \
        --data all_camp_long.csv \
        --outdir "${OUTDIR}/seed_${SEED}" \
        --maxiter 300 \
        --popsize 25 \
        --seed $SEED
done

echo ""
echo "All done. Results in: $OUTDIR"
