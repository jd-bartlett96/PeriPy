from peripy.peridynamics import assemble_K_global, find_displacements_implicit
import numpy as np
from peripy.integrators import Implicit


r = np.linspace(0, 1, 3)
r = np.array([[element] for element in r], dtype=np.float64)
r0 = r
nlist = np.array([[1, -1], [0, 2], [1, -1]], dtype=np.int32)
n_neigh = np.array([1, 2, 1], dtype=np.int32)
volume = np.array([1/4, 1/2, 1/4], dtype=np.float64)
bond_stiffness = np.float64(3)
bc_values = np.array([0, 0, 1], dtype=np.float64)
bc_types = np.array([1, 0, 1], dtype=np.float64)
displacement_bc_magnitude = 0.05


def main():
    K_global = assemble_K_global(r, nlist, n_neigh, volume, bond_stiffness,
                        bc_values, bc_types)
    
    u, x = find_displacements_implicit(K_global, r, displacement_bc_magnitude, bc_types, bc_values)
    u = [[element] for element in u]
    print(u, x)

    integrator = Implicit(None)
    # Build the integrator
    integrator.build(
        None, r, volume, None, bc_types,
        bc_values, None, None, None, None, None)
    integrator.create_buffers(
        nlist, n_neigh, bond_stiffness, 0.5, None,
        np.array([1]), None, None, None, None, None, None, 1, 1
    )
    
    integrator(displacement_bc_magnitude, None)
    print(integrator.u)
    



if __name__ == "__main__":
    main()