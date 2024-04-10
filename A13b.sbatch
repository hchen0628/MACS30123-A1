#!/bin/bash

#SBATCH --job-name=A13b
#SBATCH -o A13b.out
#SBATCH -e A13b.err
#SBATCH --partition=caslake
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hchen0628@rcc.uchicago.edu
#SBATCH --account=macs30123

module load python mpich

python3 ./A13b.py >> A13b.out
