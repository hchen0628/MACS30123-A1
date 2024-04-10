#!/bin/bash
#SBATCH --job-name=gpu_ndvi
#SBATCH --output=gpu_ndvi.out
#SBATCH --error=gpu_ndvi.err
#SBATCH --account=macs30123

for i in {1..20}
do
mpirun -n $i 123_1b_test.py >> mpi.out
done

