#!/bin/bash
#
#SBATCH --job-name=benchopt-torch
#
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate benchopt-torch

cd ~/benchmarks/benchmark-torch-distributed-linreg
python -m benchopt run . --config launch_config.yml