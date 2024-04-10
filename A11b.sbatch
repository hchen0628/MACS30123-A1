#!/bin/bash

#SBATCH --job-name=A11b
#SBATCH -o A11b.out
#SBATCH -e A11b.err
#SBATCH --partition=caslake
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hchen0628@rcc.uchicago.edu
#SBATCH --account=macs30123

for i in {1..20}
do
    mpirun -n $i A11b.py >> A11b.out
done
