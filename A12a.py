from numba.pycc import CC
import numpy as np
import scipy.stats as sts
from mpi4py import MPI
import time

cc = CC('compiled_simulate_lifetime')

@cc.export('simulate_lifetime', 'f8[:](f8[:,:], f8, f8, f8, f8, i4, i4)')
def simulate_lifetime(eps_mat, rho, mu, sigma, z_0, T, S):
    periods_to_negative = np.zeros(S)
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            if z_t <= 0:
                periods_to_negative[s_ind] = t_ind + 1
                break
            z_tm1 = z_t
    return periods_to_negative

if __name__ == "__main__":
    # Compile the numba part first
    cc.compile()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Import the compiled module
    from compiled_simulate_lifetime import simulate_lifetime

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
        results[i] = np.mean(periods_to_negative[periods_to_negative > 0])

    # Gather results from all cores
    all_results = comm.gather(results, root=0)

    # Rank 0 combines results and finds the optimal rho
    if rank == 0:
        combined_results = np.concatenate(all_results)
        optimal_rho_index = np.argmax(combined_results)
        optimal_rho = rho_values[optimal_rho_index]
        computation_time = time.time() - start_time
        print(f"Optimal ρ: {optimal_rho}, Computation time: {computation_time} seconds")
