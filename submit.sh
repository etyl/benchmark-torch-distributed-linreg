#!/bin/bash
#
#SBATCH --job-name=benchopt-torch
#
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --partition=normal,parietal,gpu-best
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate benchopt-torch

# 1. Extract the IP address of the master node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Master Node IP: $head_node_ip"

# 2. Export environment variables needed for PyTorch DDP
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500  # Any free port

cd ~/benchmarks/benchmark-torch-distributed-linreg
python -m benchopt run . --config launch_config.yml


#SBATCH --job-name=benchopt-temp
#SBATCH --partition gpu_p2
#SBATCH --nodes 2
#SBATCH --gpus-per-node 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --qos qos_gpu-dev
#SBATCH --account lzs@v100

srun python temp.py