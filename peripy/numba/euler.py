import numpy as np
from numba import njit, prange

# @njit(nogil=True, parallel=True)  # TODO: why no jit compile... vectorised?
def update_displacement(
        nodal_force, nodal_displacement, nodal_velocity,
        density, bc_type, bc_values, bc_scale, DT):
    damping = 2.5e6
    nodal_acceleration = (nodal_force - damping * nodal_velocity) / density
    index = np.where(bc_type[:, 1] == 1)  # TODO: fix this TODO shouldn't need this if boundary conditions are prescribed.
    nodal_acceleration[index, 2] = 0
    # nodal_acceleration[bc_type == 1] = 0  # Apply boundary conditions - constraints
    nodal_velocity_forward = nodal_velocity + (nodal_acceleration * DT)
    nodal_displacement_DT = nodal_velocity_forward * DT
    nodal_displacement_forward = nodal_displacement + nodal_displacement_DT
    nodal_displacement_forward[bc_values != 0] = 0  # Apply boundary conditions - applied displacements #TODO this isn't general
    nodal_displacement_forward = (nodal_displacement_forward +
                                  (bc_scale * bc_values))
    return nodal_displacement_forward, nodal_velocity_forward

