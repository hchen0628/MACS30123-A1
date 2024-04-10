from mpi4py import MPI
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import time

# Import the AOT compiled simulation function
from compiled_simulate_lifetime import simulate_lifetime

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set the random seed on rank 0 and broadcast the seed to ensure reproducibility
if rank == 0:
    seed = 0
else:
    seed = None
seed = comm.bcast(seed, root=0)
np.random.seed(seed)

# Define simulation parameters
rho_values = np.linspace(-0.95, 0.95, 200)  # 200 values of rho
mu = 3.0
sigma = 1.0
S = 1000  # Number of lives
T = 4160  # Number of periods
z_0 = mu - 3 * sigma  # Starting health index below average

# Generate shocks only once on rank 0 and broadcast to all processes
if rank == 0:
    eps_mat_global = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
else:
    eps_mat_global = np.empty((T, S), dtype=np.float64)
comm.Bcast(eps_mat_global, root=0)

# Divide rho_values evenly across processes
rho_values_subset = np.array_split(rho_values, size)[rank]

# Initialize an array to store results for each core
results = np.zeros(len(rho_values_subset))

start_time = MPI.Wtime()

# Perform simulations for the subset of rho values
for i, rho in enumerate(rho_values_subset):
    periods_to_negative = simulate_lifetime(eps_mat_global, rho, mu, sigma, z_0, T, S)
    results[i] = np.mean(periods_to_negative[periods_to_negative > 0])  # Exclude zeros

# Gather results from all processes on the root
all_results = None
if rank == 0:
    all_results = np.empty([size, len(rho_values_subset)], dtype=np.float64)
comm.Gather(results, all_results, root=0)

# Only rank 0 processes the results and generates the plot
if rank == 0:
    # Flatten and concatenate the results since the number of rho_values may not divide evenly
    combined_results = np.concatenate(all_results)[:len(rho_values)]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, combined_results, marker='o', linestyle='-')
    plt.title('Average Number of Periods to First Negative Health Index')
    plt.xlabel(r'$\rho$ Values')
    plt.ylabel('Average Number of Periods')
    plt.grid(True)
    plt.savefig('health_index_simulation_results.png')  # Save the figure
    plt.show()

    # Report computation time
    computation_time = MPI.Wtime() - start_time
    print(f"Computation time: {computation_time} seconds")
