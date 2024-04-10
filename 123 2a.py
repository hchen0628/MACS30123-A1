from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import time
# Run the simulation_model.py script to compile the module, type "simulate_lifetime.py" in terminal
from compiled_simulate_lifetime import simulate_lifetime  # Import the AOT compiled function

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Simulation parameters
rho_values = np.linspace(-0.95, 0.95, 200)  # 200 values of rho
mu = 3.0
sigma = 1.0
S = 1000  # Number of lives
T = 4160  # Number of weeks
z_0 = mu - 3*sigma  # Starting health index below average
np.random.seed(25)

# Only rank 0 generates the shocks and broadcasts to other processes
if rank == 0:
    eps_mat_global = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
else:
    eps_mat_global = np.empty((T, S), dtype=np.float64)
comm.Bcast(eps_mat_global, root=0)

# Divide the task among the cores
rho_values_subset = np.array_split(rho_values, size)[rank]

# Store results for each core
results = np.zeros(len(rho_values_subset))

start_time = time.time()

# Run simulations
for i, rho in enumerate(rho_values_subset):
    periods_to_negative = simulate_lifetime(eps_mat_global, rho, mu, sigma, z_0, T, S)
    results[i] = np.mean(periods_to_negative[periods_to_negative > 0])  # Average excluding zeros

# Gather results from all cores
all_results = comm.gather(results, root=0)

# Rank 0 combines results and finds the optimal rho
if rank == 0:
    combined_results = np.concatenate(all_results)
    optimal_rho_index = np.argmax(combined_results)
    optimal_rho = rho_values[optimal_rho_index]
    computation_time = time.time() - start_time
    print(f"Optimal ρ: {optimal_rho}, Computation time: {computation_time} seconds")

