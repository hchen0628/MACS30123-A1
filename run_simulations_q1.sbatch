#!/bin/bash

#SBATCH --job-name=run_simulations_q1
#SBATCH -o run_simulations_q1.out
#SBATCH -e run_simulations_q1.err
#SBATCH --partition=caslake
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hchen0628@rcc.uchicago.edu
#SBATCH --account=macs30123

for i in {1..20}
do
    mpirun -n $i 123_1b_test.py >> run_simulations_q1.out
done
