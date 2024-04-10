#!/bin/bash

#SBATCH --job-name=gpu_ndvi
#SBATCH --output=gpu_ndvi.out
#SBATCH --error=gpu_ndvi.err
#SBATCH --nodes=1              # Request 1 GPU node
#SBATCH --ntasks-per-node=1    # Request 1 task per GPU node
#SBATCH --partition=gpu        # Specify the GPU partition
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --account=macs30123    # Specify the account for job charging
#SBATCH --mem-per-cpu=30G      # Limit the memory usage per CPU

module load python/3.8.0

python3 "/home/hchen0628/MACS30123-A1/123_3a.py"
