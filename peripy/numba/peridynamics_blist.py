from numba.np.ufunc import parallel
import numpy as np
from numba import njit, prange
from peripy.numba.damage import (
    bond_damage_trilinear, bond_damage_PMB, bond_damage_sigmoid)
# TODO: Is math.sqrt faster than np.sqrt?
# TODO: Initialising deformed_X, _Y, _Z every iteration is expensive: try having deformed_X as input arguments using .copy()
# TODO: compare with a neighbour list as opposed to bond list (for continuity with the GPU code).


@njit(parallel=True)
def numba_node_force_blist(
        volume, bond_stiffness, sc, bond_damage, bond_force,
        global_size, blist, u, r0, node_force, force_bc_values,
        force_bc_types, force_bc_magnitude):
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
        bond_damage_temp = bond_damage_PMB(
            stretch, sc, bond_damage[global_id])
        # TODO: clearly need a better way to switch between functions
        # bond_damage_tmp = bond_damage_trilinear(
        #     stretch, sc[0], sc[1], sc[2], bond_damage[global_id], beta=0.25)
        # bond_damage = bond_damage_sigmoid(
        #     global_size, stretch, sc, sigma, bond_damage)
        bond_damage[global_id] = bond_damage_temp
        f = stretch * bond_stiffness * (
            1.0 - bond_damage_temp) * volume[node_id_j]
        bond_force[global_id, 0] = f * xi_eta_x / y
        bond_force[global_id, 1] = f * xi_eta_y / y
        bond_force[global_id, 2] = f * xi_eta_z / y
    for global_id in range(global_size):
        node_id_i = blist[global_id, 0]
        node_id_j = blist[global_id, 1]
        # Newton's 3rd law, assuming that bonds are not counted twice
        node_force[node_id_i, :] += bond_force[global_id, :]
        node_force[node_id_j, :] -= bond_force[global_id, :]
    # Neumann boundary conditions
    # (unroll catesian loop because np.where does not support 2D indexing?)
    node_force[:, 0] = np.where(
        force_bc_types[:, 0] == 0,
        node_force[:, 0],
        node_force[:, 0] + force_bc_magnitude * force_bc_values[:, 0])
    node_force[:, 1] = np.where(
        force_bc_types[:, 1] == 0,
        node_force[:, 1],
        node_force[:, 1] + force_bc_magnitude * force_bc_values[:, 1])
    node_force[:, 2] = np.where(
        force_bc_types[:, 2] == 0,
        node_force[:, 2],
        node_force[:, 2] + force_bc_magnitude * force_bc_values[:, 2])
    return node_force, bond_damage


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


# TODO: test it
# @njit(parallel=True)
# def numba_damage(global_size, blist, nnodes, bond_damage, family):
#     neighbors = np.zeros(nnodes)
#     for global_id in range(global_size):
#         node_id_i = blist[global_id, 0]
#         if bond_damage[global_id] != 1.0:
#             neighbors[node_id_i] += 1
#     return 1 - neighbors / family


# TODO: test it
# @njit
# def numba_damage(family, nnodes, blist, bond_damage):
#     unbroken_bonds = np.zeros(nnodes)
#     for k, bond in enumerate(blist):
#         node_i = bond[0]
#         node_j = bond[1]
#         unbroken_bonds[node_i] = unbroken_bonds[node_i] + bond_damage[k]
#         unbroken_bonds[node_j] = unbroken_bonds[node_j] + bond_damage[k]
#     damage = 1 - (unbroken_bonds / family)
#     return damage
