from numba import njit, prange
import numpy as np


# TODO: which is faster?
@njit(parallel=True)
def update_displacement(
    nnodes, force, u, bc_types, bc_values, displacement_bc_magnitude, dt):
    """
    The bond lengths are not precalculated. Seeing if paralellising this way,
    by merging all functions, which gets rid of overhead, is faster.
    """
    # TODO: Need to unroll 1st axis since the numpy where is cannot operate on 2D array.
    u[:, 0] = np.where(
        bc_types[:, 0] == 0, u[:, 0] + dt * force[:, 0],
        displacement_bc_magnitude * bc_values[:, 0])
    u[:, 1] = np.where(
        bc_types[:, 1] == 0, u[:, 1] + dt * force[:, 1],
        displacement_bc_magnitude * bc_values[:, 1])
    u[:, 2] = np.where(
        bc_types[:, 2] == 0, u[:, 2] + dt * force[:, 2],
        displacement_bc_magnitude * bc_values[:, 2])
    return u


# TODO: Need to unroll cartesian loop?
@njit(parallel=True)
def update_displacementss(
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
            u[node_id_i, 0] = displacement_bc_magnitude * bc_values[
                node_id_i, 0]
        if bc_types[node_id_i, 1] != 0:
            u[node_id_i, 1] = displacement_bc_magnitude * bc_values[
                node_id_i, 1]
        if bc_types[node_id_i, 2] != 0:
            u[node_id_i, 2] = displacement_bc_magnitude * bc_values[
                node_id_i, 2]
    return u
