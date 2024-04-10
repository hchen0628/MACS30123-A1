from numba.pycc import CC
import numpy as np
import scipy.stats as sts
import time

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

def simulate_health_indices_original(eps_mat, z_mat, rho, mu, z_0):
    S = z_mat.shape[1]
    T = z_mat.shape[0]
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
    return z_mat

if __name__ == "__main__":
    # Compile the numba part first
    cc.compile()

    # Import the compiled module
    from compiled_simulation import update_health_indices

    # Set model parameters and simulation parameters
    rho, mu, sigma, z_0 = 0.5, 3.0, 1.0, 3.0
    S, T = 1000, 4160
    np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
    z_mat = np.zeros((T, S))

    # Wrapper function to use the AOT compiled function
    def simulate_health_indices_aot(eps_mat, rho, mu, z_0):
        S = eps_mat.shape[1]
        T = eps_mat.shape[0]
        z_mat = np.zeros((T, S))
        update_health_indices(eps_mat, z_mat, rho, mu, z_0)
        return z_mat

    # Measure time for the original version
    start_time = time.time()
    simulate_health_indices_original(eps_mat, z_mat, rho, mu, z_0)
    original_time = time.time() - start_time

    # Measure time for the AOT-compiled Numba version
    start_time = time.time()
    simulate_health_indices_aot(eps_mat, rho, mu, z_0)
    compiled_time = time.time() - start_time

    # Calculate speedup and print results
    speedup = original_time / compiled_time
    print(f"Original execution time: {original_time} seconds")
    print(f"Numba-accelerated execution time: {compiled_time} seconds")
    print(f"Speedup observed: {speedup:.2f}x")
