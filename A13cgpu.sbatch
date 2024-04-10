#!/bin/bash

#SBATCH --job-name=A13cgpu
#SBATCH --output=A13cgpu.out
#SBATCH --error=A13cgpu.err
#SBATCH --nodes=1              # Request 1 GPU node
#SBATCH --ntasks-per-node=1    # Request 1 task per GPU node
#SBATCH --partition=gpu        # Specify the GPU partition
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --account=macs30123    # Specify the account for job charging
#SBATCH --mem-per-cpu=30G      # Limit the memory usage per CPU

module load cuda python

python3 ./A13cgpu.py >> A13cgpu.out
