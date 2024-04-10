#!/bin/bash

#SBATCH --job-name=A12c
#SBATCH -o A12c.out
#SBATCH -e A12c.err
#SBATCH --partition=caslake
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hchen0628@rcc.uchicago.edu
#SBATCH --account=macs30123

module load python mpich

mpirun -n 10 python3 ./A12c.py >> A12c.out
