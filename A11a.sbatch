#!/bin/bash

#SBATCH --job-name=A11a
#SBATCH -o A11a.out
#SBATCH -e A11a.err
#SBATCH --partition=caslake
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hchen0628@rcc.uchicago.edu
#SBATCH --account=macs30123

module load python mpich

python3 ./A11a.py >> A11a.out
