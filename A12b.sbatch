#!/bin/bash

#SBATCH --job-name=A12b
#SBATCH -o A12b.out
#SBATCH -e A12b.err
#SBATCH --partition=caslake
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hchen0628@rcc.uchicago.edu
#SBATCH --account=macs30123

module load python mpich

mpirun -n 10 python3 ./A12b.py >> A12b.out
