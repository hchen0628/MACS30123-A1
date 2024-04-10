#!/bin/bash

#SBATCH --job-name=gpu_ndvi
#SBATCH --output=gpu_ndvi.out
#SBATCH --error=gpu_ndvi.err
#SBATCH --nodes=1  # Using 1 GPU node
#SBATCH --ntasks-per-node=1  # 1 CPU node to drive the GPU
#SBATCH --partition=gpu  # Using the GPU partition on Midway3
#SBATCH --gres=gpu:1  # Requesting only one GPU
#SBATCH --account=macs30123
#SBATCH --mem-per-cpu=30G  # Restrict CPU memory usage by 30G

module load cuda python

python3 "123 3a.py"