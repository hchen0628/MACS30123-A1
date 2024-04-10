from numba.pycc import CC
import numpy as np
import scipy.stats as sts
from mpi4py import MPI

# Prepare the optimized portion using numba.pycc
cc = CC('compiled_simulation')

@cc.export('update_health_indices', 'void(f8[:,:], f8[:,:], f8, f8, f8)')
def update_health_indices(eps_mat, z_mat, rho, mu, z_0):
    S = z_mat.shape[1]
    T = z_mat.shape[0]
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

if __name__ == "__main__":
    # Compile the numba part
    cc.compile()

    # MPI code starts here
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

    # Import the compiled module
    from compiled_simulation import update_health_indices

    # Perform the simulation
    start_time = time.time()
    update_health_indices(eps_mat, z_mat, rho, mu, z_0)
    accelerated_time = time.time() - start_time

    # Gather and report results
    accelerated_times = comm.gather(accelerated_time, root=0)
    if rank == 0:
        average_time = np.mean(accelerated_times)
        print(f"Average execution time across {size} cores: {average_time} seconds")
