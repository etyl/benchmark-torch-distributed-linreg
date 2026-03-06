#!/bin/bash
#
#SBATCH --job-name=benchopt-torch
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_cpu-dev
#SBATCH --time=00:20:00


module load pytorch-gpu
python -m benchopt run . --config launch_config.yml --collect

