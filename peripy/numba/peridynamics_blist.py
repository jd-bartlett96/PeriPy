import numpy as np
from numba import njit, prange
from .damage import bond_damage_PMB, bond_damage_trilinear
# TODO: Is math.sqrt faster than np.sqrt?
# TODO: Initialising deformed_X, _Y, _Z every iteration is expensive: try having deformed_X as input arguments using .copy()
# TODO: compare with a neighbour list as opposed to bond list (for continuity with the GPU code).


# @njit
# def bond_damage_PMB(
#         stretch, sc, bond_damage):
#     """
#     Calculate the bond softening factors for the trilinear model.

#     Also known as ``bond damge'', the bond softening factors are applied to
#     satisfy the damage law.

#     :arg int global_size: The number of bonds.
#     :arg stretch:
#     :type stretch:
#     :arg float s0:
#     :arg float s1:
#     :arg float sc:
#     :arg bond_damage:
#     :type bond_damage:
#     :arg float beta:
#     """
#     # bond softening factors will not increase from 0 under linear elastic
#     # regime
#     if stretch < sc:
#         bond_damage_temp = 0.0
#     else:
#         bond_damage_temp = 1.0
#     # bond softening factor can only increase (damage is irreversible)
#     if bond_damage_temp > bond_damage:
#         bond_damage = bond_damage_temp
#     return bond_damage


@njit(parallel=True)
def numba_node_force_blist(
        volume, bond_stiffness, sc, bond_damage,
        global_size, blist, u, r0, node_force, force_bc_values,
        force_bc_types, force_bc_magnitude):
    """
    The bond lengths are not precalculated. Seeing if paralellising this way,
    by merging all functions, which gets rid of overhead, is faster.
    """
    f_x = np.zeros(global_size)
    f_y = np.zeros(global_size)
    f_z = np.zeros(global_size)

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
        # bond_damage_temp = bond_damage_PMB(
        #     stretch, sc[0], bond_damage[global_id])
        # bond_damage = bond_damage_sigmoid(
        #     global_size, stretch, sc, sigma, bond_damage)
        s0 = 1.05e-4
        s1 = 6.90e-4
        sc = 5.56e-3
        beta = 0.25
        bond_damage[global_id] = bond_damage_trilinear(
            stretch, s0, s1, sc, bond_damage[global_id], beta)
        # bond_damage[global_id] = bond_damage_temp
        # TODO: bond_stiffness should not be an array
        f = stretch * bond_stiffness[0] * (
            1 - bond_damage[global_id]) * volume[node_id_j]
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
    node_force[:, 0] = np.where(force_bc_types[:, 0] == 0, node_force[:, 0], node_force[:, 0] + force_bc_magnitude * force_bc_values[:, 0])
    node_force[:, 1] = np.where(force_bc_types[:, 1] == 0, node_force[:, 1], node_force[:, 1] + force_bc_magnitude * force_bc_values[:, 1])
    node_force[:, 2] = np.where(force_bc_types[:, 2] == 0, node_force[:, 2], node_force[:, 2] + force_bc_magnitude * force_bc_values[:, 2])
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

