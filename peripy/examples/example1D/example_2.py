from peripy.peridynamics import assemble_K_global, assemble_C_matrix, find_delta_u
import numpy as np


r = np.linspace(0, 1, 3)
r = np.array([[element] for element in r], dtype=np.float64)
r0 = r
n_list = np.array([[1, -1], [0, 2], [1, -1]], dtype=np.int32)
n_neigh = np.array([1, 2, 1], dtype=np.int32)
volume = np.array([1/4, 1/2, 1/4], dtype=np.float64)
bond_stiffness = np.float64(3)
bc_values = np.array([0, 0, 1], dtype=np.float64)
bc_types = np.array([1, 0, 1], dtype=np.float64)
displacement_bc_magnitude = 0.0


def main():
    K_reduced, K_global, C = assemble_K_global(r, r0, n_list, n_neigh, volume, bond_stiffness,
                        bc_values, bc_types)
    
    u = find_delta_u(K_reduced, K_global, C, r, displacement_bc_magnitude, bc_types, bc_values)
    print(u)



if __name__ == "__main__":
    main()

