from numba import njit
from numba import njit, prange

# #@njit(nogil=True, parallel=True)
# def update_displacement(
#         force, u, bc_indices, bc_values, displacement_bc_magnitude, dt):
#     """
#     Calculate the displacement of each node using an Euler
#     integrator.
#     :arg force: An (n,3) array of the forces of each node.
#     :type force:
#     :arg u: An (n,3) array of the current displacements of each node.
#     :type u:
#     :arg bc_types: An (n,3) array of the boundary condition types.
#     :type bc_types:
#     :arg bc_values: An (n,3) array of the boundary condition values applied to
#         the nodes.
#     :type bc_values:
#     :arg bc_scale: The scalar value applied to the displacement BCs.
#     :type bc_scale:
#     :arg dt: The time step in [s].
#     """
#     u = u + dt * force
#     u[bc_indices] = displacement_bc_magnitude * bc_values[bc_indices]
#     return u

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
