#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --partition=normal,parietal,gpu-best
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G

# Load necessary modules (adjust these to your cluster's specific modules)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate benchopt-torch

# Activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate

# 1. Extract the IP address of the master node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Master Node IP: $head_node_ip"

# 2. Export environment variables needed for PyTorch DDP
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500  # Any free port

# Optional: Disable NCCL P2P if you encounter hanging issues on certain cluster setups
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# 3. Launch the script using srun
# srun will automatically launch 'ntasks-per-node' * 'nodes' processes.
srun python test.py