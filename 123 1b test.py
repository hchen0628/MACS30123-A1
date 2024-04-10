from mpi4py import MPI
import numpy as np
import scipy.stats as sts

# Run the simulation_model.py script to compile the module, type "python simulation_model.py" in terminal
from compiled_simulation import update_health_indices  # Assuming this is the AOT compiled function

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Model parameters
rho, mu, sigma, z_0 = 0.5, 3.0, 1.0, 3.0

# Simulation parameters adjusted for parallel execution
S, T = 1000 // size, 4160

# Different seeds for each process
np.random.seed(rank)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
z_mat = np.zeros((T, S))

# Perform the simulation
start_time = MPI.Wtime()
z_mat_accelerated = update_health_indices(eps_mat, np.zeros((T, S)), rho, mu, z_0)
accelerated_time = MPI.Wtime() - start_time

# Gather and report results
accelerated_times = comm.gather(accelerated_time, root=0)
if rank == 0:
    average_time = np.mean(accelerated_times)
    print(f"Average execution time across {size} cores: {average_time} seconds")