from numba.pycc import CC
import numpy as np

cc = CC('compiled_simulation')

# The optimized portion as a separate function
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
    cc.compile()
