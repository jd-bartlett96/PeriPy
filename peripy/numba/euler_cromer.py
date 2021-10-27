from numba import njit

@njit(nogil=True, parallel=True)
def update_displacement(
        force, u, ud, bc_indices, bc_values, bc_scale, dt, damping, densities):
    """
    Calculate the displacement of each node using an Euler
    integrator.
    :arg force: An (n,3) array of the forces of each node.
    :type force:
    :arg u: An (n,3) array of the current displacements of each node.
    :type u:
    :arg ud: An (n,3) array of the current velocities of each node.
    :type ud:
    :arg bc_indices: An array of indices for fancy indexing condition types.
    :type bc_indices:
    :arg bc_values: An (n,3) array of the boundary condition values applied to
        the nodes.
    :type bc_values:
    :arg bc_scale: The scalar value applied to the displacement BCs.
    :type bc_scale:
    :arg dt: The time step in [s].
    :returns: The displacements, velocities and accelerations of each node.
    """
    udd = (force - damping * ud) / densities
    ud += ud * dt
    u = u + dt * ud
    u[bc_indices] = bc_scale * bc_values[bc_indices]
    return u, ud, udd
