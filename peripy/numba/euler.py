from numba import njit

@njit(nogil=True, parallel=True)
def update_displacement(
        force, u, bc_indices, bc_values, bc_scale, dt):
    """
    Calculate the displacement of each node using an Euler
    integrator.
    :arg force: An (n,3) array of the forces of each node.
    :type force:
    :arg u: An (n,3) array of the current displacements of each node.
    :type u:
    :arg bc_types: An (n,3) array of the boundary condition types.
    :type bc_types:
    :arg bc_values: An (n,3) array of the boundary condition values applied to
        the nodes.
    :type bc_values:
    :arg bc_scale: The scalar value applied to the displacement BCs.
    :type bc_scale:
    :arg dt: The time step in [s].
    """
    u = u + dt * force
    u[bc_indices] = bc_scale * bc_values[bc_indices]
    return u
