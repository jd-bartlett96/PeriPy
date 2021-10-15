from peripy.peridynamics import assemble_K_global
import numpy as np


r = np.linspace(0, 1, 3)
r = np.array([[element] for element in r], dtype=np.float64)
r0 = r
n_list = np.array([[0, 1, 2], [1,1,1]], dtype=np.float64)
n_neigh = np.array([1, 1, 1], dtype=np.float64)
volume = np.array([1/3, 1/3, 1/3], dtype=np.float64)
bond_stiffness = np.float64(3)
bc_values = np.array([0, 1, 1], dtype=np.float64)
bc_types = np.array([1, 0, 1], dtype=np.float64)

#print(type(r[0][0]), type(r0[0][0]), type(n_list[0][0]), type(n_neigh[0]))

def main():
    K, C = assemble_K_global(r, r0, n_list, n_neigh, volume, bond_stiffness,
                        bc_values)


if __name__ == "__main__":
    main()

