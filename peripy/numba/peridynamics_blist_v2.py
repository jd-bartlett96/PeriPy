"""In these functions, the initial bond length has been precalculated."""
import numpy as np
from numba import njit, prange
# TODO: Is math.sqrt faster than np.sqrt?
# TODO: Initialising deformed_X, _Y, _Z every iteration is expensive: try having deformed_X as input arguments using .copy()
# TODO: compare with a neighbour list as opposed to bond list (for continuity with the GPU code).


@njit(parallel=True)
def bond_length_blist(global_size, blist, r0, l0):
    """
    Precalculate the bond lengths.
    """
    for global_id in prange(global_size):
        node_id_i = blist[global_id, 0]
        node_id_j = blist[global_id, 1]
        xi_x = r0[node_id_j, 0] - r0[node_id_i, 0]
        xi_y = r0[node_id_j, 1] - r0[node_id_i, 1]
        xi_z = r0[node_id_j, 2] - r0[node_id_i, 2]
        l0[global_id] = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
    return l0


@njit
def reduction_blist(
        nnodes, blist, bond_force_x, bond_force_y, bond_force_z):
    node_force = np.zeros((nnodes, 3), dtype=np.float64)
    for k, bond in enumerate(blist):
        node_i = bond[0]
        node_j = bond[1]
        # x-component
        node_force[node_i, 0] += bond_force_x[k]
        node_force[node_j, 0] -= bond_force_x[k]
        # y-component
        node_force[node_i, 1] += bond_force_y[k]
        node_force[node_j, 1] -= bond_force_y[k]
        # z-component
        node_force[node_i, 2] += bond_force_z[k]
        node_force[node_j, 2] -= bond_force_z[k]
    return node_force


@njit(parallel=True)
def numba_stretch(global_size, blist, u, r0, l0):
    """
    The bond lengths are precalculated.
    """
    # deformed coordinates
    r = u + r0
    # containers for stretches in cartesian directions
    xi_eta_x = np.zeros(global_size, dtype=np.float64)
    xi_eta_y = np.zeros(global_size, dtype=np.float64)
    xi_eta_z = np.zeros(global_size, dtype=np.float64)
    # for each bond in blist
    for global_id in prange(global_size):
        node_id_i = blist[global_id, 0]
        node_id_j = blist[global_id, 1]
        xi_eta_x[global_id] = r[node_id_j, 0] - r[node_id_i, 0]
        xi_eta_y[global_id] = r[node_id_j, 1] - r[node_id_i, 1]
        xi_eta_z[global_id] = r[node_id_j, 2] - r[node_id_i, 2]
    y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
    return xi_eta_x, xi_eta_y, xi_eta_z, y, (y - l0) / l0


@njit(nogil=True, parallel=True)
def numba_bond_force(
        bond_stiffness, bond_damage, stretch, volume,
        xi_eta_x, xi_eta_y, xi_eta_z, l):
    bond_force_x = (bond_stiffness * (1 - bond_damage) * stretch
                    * volume * (xi_eta_x / l))
    bond_force_y = (bond_stiffness * (1 - bond_damage) * stretch
                    * volume * (xi_eta_y / l))
    bond_force_z = (bond_stiffness * (1 - bond_damage) * stretch
                    * volume * (xi_eta_z / l))
    return bond_force_x, bond_force_y, bond_force_z


@njit
def numba_reduce_force(
    node_force, blist, bond_force_x, bond_force_y, bond_force_z,
    force_bc_types, force_bc_values, force_bc_magnitude):
    for k, bond in enumerate(blist):
        node_i = bond[0]
        node_j = bond[1]
        # x-component
        node_force[node_i, 0] += bond_force_x[k]
        node_force[node_j, 0] -= bond_force_x[k]
        # y-component
        node_force[node_i, 1] += bond_force_y[k]
        node_force[node_j, 1] -= bond_force_y[k]
        # z-component
        node_force[node_i, 2] += bond_force_z[k]
        node_force[node_j, 2] -= bond_force_z[k]
    # Neumann boundary conditions
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
    return node_force


@njit(parallel=True)
def bond_damage_PMB_2(
        global_size, stretch, sc, bond_damage):
    """
    Calculate the bond softening factors for the PMB model.
    Also known as ``bond damge'', the bond softening factors are applied to
    satisfy the damage law.
    :arg int global_size: The number of bonds.
    :arg stretch:
    :type stretch:
    :arg float sc:
    :arg bond_damage:
    :type bond_damage:
    """
    for bond in prange(global_size):
        # Factor out indexing
        stretch_bond = stretch[bond]
        # bond softening factors will not increase from 0 under linear elastic
        # loading, stretch[bond] <= s0
        bond_damage_temp = 0.0
        if stretch_bond < sc:
            bond_damage_temp = 0.0
        else:
            bond_damage_temp = 1.0
        # Bond softening factor can only increase (damage is irreversible)
        if bond_damage_temp > bond_damage[bond]:
            bond_damage[bond] = bond_damage_temp
    return bond_damage


# TODO: these may be useful later
# @njit(parallel=True)
# def bond_damage_trilinear_v2(
#         global_size, stretch, s0, s1, sc,
#         bond_damage, beta):
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
#     eta = s1 / s0
#     for bond in prange(global_size):
#         # Factor out indexing
#         stretch_bond = stretch[bond]
#         # bond softening factors will not increase from 0 under linear elastic
#         # loading, stretch[bond] <= s0
#         if (stretch_bond > s0) and (stretch_bond <= s1):
#             bond_damage_temp = (
#                 1 - ((eta - beta) / (eta - 1) * (s0 / stretch_bond))
#                 + ((1 - beta) / (eta - 1)))
#         elif (stretch_bond > s1) and (stretch_bond <= sc):
#             bond_damage_temp = 1 - (
#                     (s0 * beta / stretch_bond)
#                     * ((sc - stretch_bond) / (sc - s1)))
#         elif stretch_bond > sc:
#             bond_damage_temp = 1
#         # Bond softening factor can only increase (damage is irreversible)
#         if bond_damage_temp > bond_damage[bond]:
#             bond_damage[bond] = bond_damage_temp
#     return bond_damage


# @njit
# def bond_damage_exponential_v2(
#         global_size, stretch, s0, sc, bond_damage, k, alpha):
#     """
#     Calculate the bond softening factors for the trilinear model.
#     Also known as ``bond damge'', the bond softening factors are applied to
#     satisfy the damage law.
#     :arg int global_size: The number of bonds.
#     :arg stretch:
#     :type stretch:
#     :arg float s0:
#     :arg float sc:
#     :arg bond_damage:
#     :type bond_damage:
#     :arg float k:
#     :arg float alpha:
#     """
#     for bond in range(global_size):
#         stretch_bond = stretch[bond]
#         if (stretch_bond > s0) and (stretch_bond < sc):
#             numerator = 1 - np.exp(-k * (stretch_bond - s0) / (sc - s0))
#             residual = alpha * (1 - (stretch_bond - s0) / (sc - s0))
#             bond_damage_temp = 1 - (s0 / stretch_bond) * (
#                 (1 - numerator / (1 - np.exp(-k))) + residual) / (1 + alpha)
#         elif stretch_bond > sc:
#             bond_damage_temp = 1
#         # Bond softening factor can only increase (damage is irreversible)
#         if bond_damage_temp > bond_damage[bond]:
#             bond_damage[bond] = bond_damage_temp
#     return bond_damage
