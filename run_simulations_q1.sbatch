1 #!/bin/bash
2
3 #SBATCH --job-name=<run_simulations_q1>
4 #SBATCH -o <run_simulations_q1>.out
5 #SBATCH -e <run_simulations_q1>.err
6 #SBATCH --partition=caslake
7 #SBATCH --ntasks=20
8 #SBATCH --nodes=1
9 #SBATCH --mail-type=ALL
10 #SBATCH --mail-user=hchen0628@rcc.uchicago.edu
11 #SBATCH --account=macs30123
12
13 module load python mpich
14
15 for i in {1..20}
16 mpirun -n $i python3 123_1b_test.py.py
17 done
