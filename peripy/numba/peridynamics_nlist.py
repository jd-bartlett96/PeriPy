import numpy as np
from numba import njit, prange
from .damage import bond_damage_PMB, bond_damage_trilinear
# TODO: Is math.sqrt faster than np.sqrt?
# TODO: Initialising deformed_X, _Y, _Z every iteration is expensive
# TODO: then have deformed_X as input arguments using .copy()


@njit(parallel=True)
def bond_length_nlist(nnodes, max_neigh, nlist, r0, l0):
    """
    Precalculate the bond lengths.
    """
    for node_id_i in prange(nnodes):
        for j in range(max_neigh):
            node_id_j = nlist[node_id_i, j]
            if node_id_j != -1:
                xi_x = r0[node_id_j, 0] - r0[node_id_i, 0]
                xi_y = r0[node_id_j, 1] - r0[node_id_i, 1]
                xi_z = r0[node_id_j, 2] - r0[node_id_i, 2]
                l0[node_id_i, node_id_j] = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
    return l0


@njit(parallel=True)
def numba_node_force_nlist(
        volume, bond_stiffness, sc, bond_damage,
        nnodes, nlist, u, r0, node_force, max_neigh,
        force_bc_values, force_bc_types, force_bc_magnitude,
        f_x, f_z, f_y):
    """
    The bond lengths are not precalculated. Seeing if paralellising this way,
    by merging all functions, which gets rid of overhead, is faster.
    """
    # For each node, i
    for node_id_i in prange(nnodes):
        for j in range(max_neigh):
            # Access node within node_id_i's horizon with corresponding
            # node_id_j
            node_id_j = nlist[node_id_i, j]
            # If bond is not broken
            if (node_id_j != -1) and (node_id_i < node_id_j):
                xi_x = r0[node_id_j, 0] - r0[node_id_i, 0]
                xi_y = r0[node_id_j, 1] - r0[node_id_i, 1]
                xi_z = r0[node_id_j, 2] - r0[node_id_i, 2]
                xi_eta_x = u[node_id_j, 0] - u[node_id_i, 0] + xi_x
                xi_eta_y = u[node_id_j, 1] - u[node_id_i, 1] + xi_y
                xi_eta_z = u[node_id_j, 2] - u[node_id_i, 2] + xi_z
                xi = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
                y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
                stretch = (y - xi) / xi
                # TODO: A way to switch out different damage laws might be with a lambda function or factory or kwarg
                # bond_damage[node_id_i, j] = bond_damage_PMB(
                #     stretch, 1.05e-4, bond_damage[node_id_i, j])
                s0 = 1.05e-4
                s1 = 6.90e-4
                sc = 5.56e-3
                beta = 0.25
                bond_damage[node_id_i, j] = bond_damage_trilinear(
                    stretch, s0, s1, sc, bond_damage[node_id_i, j], beta)
                # TODO: bond_stiffness should not be an array
                f = (stretch * bond_stiffness[0]
                     * (1 - bond_damage[node_id_i, j]) * volume[node_id_j])
                f_x[node_id_i, j] = f * xi_eta_x / y
                f_y[node_id_i, j] = f * xi_eta_y / y
                f_z[node_id_i, j] = f * xi_eta_z / y

    for node_id_i in range(nnodes):
        for j in range(max_neigh):
            node_id_j = nlist[node_id_i, j]
            if (node_id_j != -1) and (node_id_i < node_id_j):
                # Add force to particle node_id_i, using Newton's third law
                # subtract force from node_id_j
                node_force[node_id_i, 0] += f_x[node_id_i, j]
                node_force[node_id_j, 0] -= f_x[node_id_i, j]
                node_force[node_id_i, 1] += f_y[node_id_i, j]
                node_force[node_id_j, 1] -= f_y[node_id_i, j]
                node_force[node_id_i, 2] += f_z[node_id_i, j]
                node_force[node_id_j, 2] -= f_z[node_id_i, j]

    # Neumann boundary conditions
    node_force[:, 0] = np.where(force_bc_types[:, 0] == 0, node_force[:, 0],
                                node_force[:, 0] + force_bc_magnitude * force_bc_values[:, 0])
    node_force[:, 1] = np.where(force_bc_types[:, 1] == 0, node_force[:, 1],
                                node_force[:, 1] + force_bc_magnitude * force_bc_values[:, 1])
    node_force[:, 2] = np.where(force_bc_types[:, 2] == 0, node_force[:, 2],
                                node_force[:, 2] + force_bc_magnitude * force_bc_values[:, 2])
    return node_force, bond_damage


@njit
def numba_damage_nlistSS(nnodes, nlist, max_neigh, family):
    neighbors = np.zeros(nnodes)
    for node_id_i in prange(nnodes):
        for j in range(max_neigh):
            n_neigh = 0.0
            if nlist[node_id_i, j] != -1:
                n_neigh[node_id_i] += 1.0
    return 1 - neighbors / family


@njit
def numba_damage_nlist(nnodes, bond_damage, max_neigh, family):
    damage = np.zeros(nnodes)
    for node_id_i in prange(nnodes):
        for j in range(max_neigh):
            damage[node_id_i] += bond_damage[node_id_i, j]
    return damage / family
