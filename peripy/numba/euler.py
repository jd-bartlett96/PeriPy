from numba import njit
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def update_displacement(nnodes, degrees_freedom, force, u, bc_types,
                        bc_values, displacement_bc_magnitude, dt):
    """
    The bond lengths are not precalculated. Seeing if paralellising this way,
    by merging all functions, which gets rid of overhead, is faster.
    """
    for node_id_i in prange(nnodes):
        for dof in range(degrees_freedom):
            u[node_id_i, dof] = u[node_id_i, dof] + dt * force[node_id_i, dof]

            if bc_types[node_id_i, dof] != 0:
                u[node_id_i, dof] = (displacement_bc_magnitude
                                     * bc_values[node_id_i, dof])

    return u
