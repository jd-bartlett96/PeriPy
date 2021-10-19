import numpy as np
from numba import njit, prange
from numpy import jnp
# TODO: Is math.sqrt faster than np.sqrt?
# TODO: Initialising deformed_X, _Y, _Z every iteration is expensive
# TODO: then have deformed_X as input arguments using .copy()


@njit(parallel=True)
def calculate_stretch_0(global_size, bond_list, u, r0, l0):
    """
    The bond lengths are precalculated.
    """
    # deformed coordinates
    r = u + r0
    # containers for stretches in cartesian directions
    xi_eta_x = np.zeros(global_size)
    xi_eta_y = np.zeros(global_size)
    xi_eta_z = np.zeros(global_size)
    # for each bond in bondlist
    for global_id in prange(global_size):
        node_id_i = bond_list[global_id, 0]
        node_id_j = bond_list[global_id, 1]
        xi_eta_x[global_id] = r[node_id_j, 0] - r[node_id_i, 0]
        xi_eta_y[global_id] = r[node_id_j, 0] - r[node_id_i, 0]
        xi_eta_z[global_id] = r[node_id_j, 0] - r[node_id_i, 0]
    y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
    stretch = (y - l0) / l0
    return stretch


@njit(parallel=True)
def calculate_stretch_1(global_size, bondlist, u, r0):
    """
    The bond lengths are not precalculated. Seeing if paralellising this way,
    similar to as done on a GPU is faster.
    """
    stretch = np.empty(global_size)
    for global_id in prange(global_size):
        node_id_i = bondlist[global_id, 0]
        node_id_j = bondlist[global_id, 1]
        xi_x = r0[node_id_j, 0] - r0[node_id_i, 0]
        xi_y = r0[node_id_j, 1] - r0[node_id_i, 1]
        xi_z = r0[node_id_j, 2] - r0[node_id_i, 2]
        xi_eta_x = u[node_id_j, 0] - u[node_id_i, 0] + xi_x
        xi_eta_y = u[node_id_j, 0] - u[node_id_i, 0] + xi_x
        xi_eta_z = u[node_id_j, 0] - u[node_id_i, 0] + xi_x
        xi = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
        y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
        stretch[global_id] = (y - xi) / xi
    return stretch


@njit(parallel=True)
def calculate_stretch_2(bondlist, deformed_coordinates, bond_length):
    nbonds = len(bondlist)
    deformed_X = np.zeros(nbonds)
    deformed_Y = np.zeros(nbonds)
    deformed_Z = np.zeros(nbonds)
    for kBond in prange(nbonds):
        node_i = bondlist[kBond, 0]
        node_j = bondlist[kBond, 1]
        deformed_X[kBond] = (
            deformed_coordinates[node_j, 0] - deformed_coordinates[node_i, 0])
        deformed_Y[kBond] = (
            deformed_coordinates[node_j, 1] - deformed_coordinates[node_i, 1])
        deformed_Z[kBond] = (
            deformed_coordinates[node_j, 2] - deformed_coordinates[node_i, 2])
    deformed_length = np.sqrt(
        deformed_X ** 2 + deformed_Y ** 2 + deformed_Z ** 2)
    stretch = (deformed_length - bond_length) / bond_length
    return deformed_X, deformed_Y, deformed_Z, deformed_length, stretch



@njit(parallel=True)
def calculate_bond_softening_factor_sigmoid(
        global_size, stretch, sc, sigma, bond_softening_factor):
    """
    Calculate the bond softening factors for the sigmoid model.

    Also known as ``bond damge'', the bond softening factors are applied to
    satisfy the damage law.
    """
    for bond in prange(global_size):
        bond_softening_factor[bond] = 1 / (
            np.exp((stretch[bond] - sc) / sigma) + 1)
    return bond_softening_factor


@njit(parallel=True)
def calculate_bond_softening_factor_trilinear(
        global_size, stretch, s0, s1, sc,
        bond_softening_factor, flag_bond_softening_factor, beta=0.25):
    """
    Calculate the bond softening factors for the trilinear model.

    Also known as ``bond damge'', the bond softening factors are applied to
    satisfy the damage law.

    :arg int global_size: The number of bonds.
    :arg stretch:
    :type stretch:
    :arg float s0:
    :arg float s1:
    :arg float sc:
    :arg bond_softening_factor:
    :type bond_softening_factor:
    :arg float beta:
    """
    eta = s1 / s0
    for bond in prange(global_size):
        # Factor out indexing
        stretch_bond = stretch[bond]
        # bond softening factors will not increase from 0 under linear elastic
        # loading, stretch[bond] <= s0
        if (stretch_bond > s0) and (stretch_bond <= s1):
            flag_bond_softening_factor[bond] = 1
            bond_softening_factor_temp = (
                1 - ((eta - beta) / (eta - 1) * (s0 / stretch_bond))
                + ((1 - beta) / (eta - 1)))
        elif (stretch_bond > s1) and (stretch_bond <= sc):
            bond_softening_factor_temp = 1 - (
                    (s0 * beta / stretch_bond)
                    * ((sc - stretch_bond) / (sc - s1)))
        elif stretch_bond > sc:
            bond_softening_factor_temp = 1
        # Bond softening factor can only increase (damage is irreversible)
        if bond_softening_factor_temp > bond_softening_factor[bond]:
            bond_softening_factor[bond] = bond_softening_factor_temp
    return bond_softening_factor, flag_bond_softening_factor


@njit
def calculate_bsf_non_linear(stretch, s0, sc, bond_softening_factor, flag_bsf):
    nbonds = len(stretch)
    k = 25
    alpha = 0.25
    bsf = np.zeros(nbonds)
    for kBond in range(nbonds):
        if (stretch[kBond] > s0) and (stretch[kBond] < sc):
            numerator = 1 - np.exp(-k * (stretch[kBond] - s0) / (sc - s0))
            residual = alpha * (1 - (stretch[kBond] - s0) / (sc - s0))
            bsf[kBond] = 1 - (s0 / stretch[kBond]) * (
                (1 - numerator / (1 - np.exp(-k))) + residual) / (1 + alpha)
            flag_bsf[kBond] = 1
        elif stretch[kBond] > sc:
            bsf[kBond] = 1
        # Bond softening factor can only increase (damage is irreversible)
        if bsf[kBond] > bond_softening_factor[kBond]:
            bond_softening_factor[kBond] = bsf[kBond]
    return bond_softening_factor, flag_bsf


@njit(nogil=True, parallel=True)
def calculate_bond_force(
    bond_stiffness, bond_softening_factor, stretch, volume,
    deformed_X, deformed_Y, deformed_Z, deformed_length):
    bond_force_X = (bond_stiffness * (1 - bond_softening_factor) * stretch
                    * volume * (deformed_X / deformed_length))
    bond_force_Y = (bond_stiffness * (1 - bond_softening_factor) * stretch
                    * volume * (deformed_Y / deformed_length))
    bond_force_Z = (bond_stiffness * (1 - bond_softening_factor) * stretch
                    * volume * (deformed_Z / deformed_length))
    return bond_force_X, bond_force_Y, bond_force_Z


@njit
def calculate_nodal_force(
    nnodes, bondlist, bond_force_X, bond_force_Y, bond_force_Z):
    nodal_force = np.zeros((nnodes, 3), dtype=np.float64)
    for kBond, bond in enumerate(bondlist):
        node_i = bond[0]
        node_j = bond[1]
        # x-component
        nodal_force[node_i, 0] += bond_force_X[kBond]
        nodal_force[node_j, 0] -= bond_force_X[kBond]
        # y-component
        nodal_force[node_i, 1] += bond_force_Y[kBond]
        nodal_force[node_j, 1] -= bond_force_Y[kBond]
        # z-component
        nodal_force[node_i, 2] += bond_force_Z[kBond]
        nodal_force[node_j, 2] -= bond_force_Z[kBond]
    return nodal_force


# @njit(nogil=True, parallel=True)  # TODO: why no jit compile... vectorised?
def euler_cromer(nodal_force, nodal_displacement, nodal_velocity, density,
                 bc_type, bc_values, bc_scale, DT):
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


@njit
def calculate_damage(n_family_members, bondlist, fail):
    nnodes = len(n_family_members)
    unbroken_bonds = np.zeros(nnodes)
    for kBond, bond in enumerate(bondlist):
        node_i = bond[0]
        node_j = bond[1]
        unbroken_bonds[node_i] = unbroken_bonds[node_i] + fail[kBond]
        unbroken_bonds[node_j] = unbroken_bonds[node_j] + fail[kBond]
    damage = 1 - (unbroken_bonds / n_family_members)
    return damage
