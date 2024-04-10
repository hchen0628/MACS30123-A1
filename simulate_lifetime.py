from numba.pycc import CC
import numpy as np

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
    cc.compile()
