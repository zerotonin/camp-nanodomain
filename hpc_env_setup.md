# HPC env setup
conda create -n camp-nanodomain python=3.11 numpy scipy pandas matplotlib tqdm -c conda-forge -y
conda activate camp-nanodomain
pip install "jax[cuda12]==0.4.30" "diffrax>=0.5.0,<0.7" "equinox>=0.11,<0.12" "optax>=0.2.2"

# Copy files to cluster
scp optimise_model_gpu_v2.py all_camp_long.csv slurm_camp_optimise.sh <cluster>:<project_dir>/

# Quick test (50 generations, ~15 min on H100)
sbatch --time=00:30:00 --export=ALL,QUICK=1 slurm_camp_optimise.sh

# Full run (300 generations, ~1-1.5h on H100)
sbatch slurm_camp_optimise.sh