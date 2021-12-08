import numpy as np
from numba import njit, prange
from .damage import bond_damage_trilinear
# TODO: Is math.sqrt faster than np.sqrt?
# TODO: Initialising deformed_X, _Y, _Z every iteration is expensive: try having deformed_X as input arguments using .copy()
# TODO: compare with a neighbour list as opposed to bond list (for continuity with the GPU code).


@njit(parallel=True)
def numba_node_force_blist(
        volume, bond_stiffness, sc, bond_damage,
        global_size, blist, u, r0, node_force, force_bc_values,
        force_bc_types, force_bc_magnitude, f_x, f_y, f_z):
    """
    The bond lengths are not precalculated. Seeing if paralellising this way,
    by merging all functions, which gets rid of overhead, is faster.
    """

    for global_id in prange(global_size):

        node_id_i = blist[global_id, 0]
        node_id_j = blist[global_id, 1]

        xi_x = r0[node_id_j, 0] - r0[node_id_i, 0]
        xi_y = r0[node_id_j, 1] - r0[node_id_i, 1]
        xi_z = r0[node_id_j, 2] - r0[node_id_i, 2]

        xi_eta_x = u[node_id_j, 0] - u[node_id_i, 0] + xi_x
        xi_eta_y = u[node_id_j, 1] - u[node_id_i, 1] + xi_y
        xi_eta_z = u[node_id_j, 2] - u[node_id_i, 2] + xi_z

        xi = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
        y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
        stretch = (y - xi) / xi

        s0 = 1.05e-4
        s1 = 6.90e-4
        sc = 5.56e-3
        beta = 0.25
        bond_damage[global_id] = bond_damage_trilinear(
            stretch, s0, s1, sc, bond_damage[global_id], beta)

        # TODO: bond_stiffness should not be an array
        f = (stretch * bond_stiffness[0] * (1 - bond_damage[global_id])
             * volume[node_id_j])
        f_x[global_id] = f * xi_eta_x / y
        f_y[global_id] = f * xi_eta_y / y
        f_z[global_id] = f * xi_eta_z / y

    # TODO: nodal forces can't be reduced in parallel
    for global_id in range(global_size):

        node_id_i = blist[global_id, 0]
        node_id_j = blist[global_id, 1]

        node_force[node_id_i, 0] += f_x[global_id]
        node_force[node_id_j, 0] -= f_x[global_id]
        node_force[node_id_i, 1] += f_y[global_id]
        node_force[node_id_j, 1] -= f_y[global_id]
        node_force[node_id_i, 2] += f_z[global_id]
        node_force[node_id_j, 2] -= f_z[global_id]

    # Neumann boundary conditions
    node_force[:, 0] = np.where(force_bc_types[:, 0] == 0, node_force[:, 0],
                                node_force[:, 0] + force_bc_magnitude * force_bc_values[:, 0])
    node_force[:, 1] = np.where(force_bc_types[:, 1] == 0, node_force[:, 1],
                                node_force[:, 1] + force_bc_magnitude * force_bc_values[:, 1])
    node_force[:, 2] = np.where(force_bc_types[:, 2] == 0, node_force[:, 2],
                                node_force[:, 2] + force_bc_magnitude * force_bc_values[:, 2])

    return node_force, bond_damage


# TODO: double check this function
@njit(parallel=True)
def numba_damage(global_size, blist, nnodes, bond_damage, family):
    """
    Calculate the damage of each node.
    """
    damage = np.zeros(nnodes)
    for global_id in range(global_size):
        node_id_i = blist[global_id, 0]
        damage[node_id_i] += bond_damage[global_id]

    return damage / family


# TODO: This is for some reason failing
@njit
def numba_damageSS(global_size, blist, nnodes, bond_damage, family):
    neighbors = np.zeros(nnodes)
    for global_id in prange(global_size):
        node_id_i = blist[global_id, 0]
        if bond_damage[global_id] != 1.0:
            neighbors[node_id_i] += 1
    # print(neighbors)
    # print(family)
    return 1 - neighbors / family

