# HPC env setup
conda create -n camp-nanodomain python=3.11 numpy scipy pandas matplotlib tqdm -c conda-forge -y
conda activate camp-nanodomain
pip install "jax[cuda12]==0.4.30" "diffrax>=0.5.0,<0.7" "equinox>=0.11,<0.12" "optax>=0.2.2"
