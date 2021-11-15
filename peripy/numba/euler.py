from numba import njit
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def update_displacement(
        nnodes, force, u, bc_types, bc_values, displacement_bc_magnitude, dt):
    """
    The bond lengths are not precalculated. Seeing if paralellising this way,
    by merging all functions, which gets rid of overhead, is faster.
    """
    for node_id_i in prange(nnodes):
        u[node_id_i, 0] = u[node_id_i, 0] + dt * force[node_id_i, 0]
        u[node_id_i, 1] = u[node_id_i, 1] + dt * force[node_id_i, 1]
        u[node_id_i, 2] = u[node_id_i, 2] + dt * force[node_id_i, 2]
        if bc_types[node_id_i, 0] != 0:
            u[node_id_i, 0] = displacement_bc_magnitude * bc_values[node_id_i, 0]
        if bc_types[node_id_i, 1] != 0:
            u[node_id_i, 1] = displacement_bc_magnitude * bc_values[node_id_i, 1]
        if bc_types[node_id_i, 2] != 0:
            u[node_id_i, 2] = displacement_bc_magnitude * bc_values[node_id_i, 2]
    return u
