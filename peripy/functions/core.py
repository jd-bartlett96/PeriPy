import numpy
from numba import njit


# --------------------------------------------
#           Calculate stretch
# --------------------------------------------
# TODO: Is math.sqrt faster than numpy.sqrt?
# TODO: Initialising deformed_X, _Y, _Z every iteration is expensive

@njit
def calculate_stretch(bondlist, deformed_coordinates, bond_length):

    nbonds = len(bondlist)
    deformed_X = numpy.zeros(nbonds)
    deformed_Y = numpy.zeros(nbonds)
    deformed_Z = numpy.zeros(nbonds)

    for kBond, bond in enumerate(bondlist):
        node_i = bond[0]
        node_j = bond[1]

        deformed_X[kBond] = deformed_coordinates[node_j, 0] - deformed_coordinates[node_i, 0]
        deformed_Y[kBond] = deformed_coordinates[node_j, 1] - deformed_coordinates[node_i, 1]
        deformed_Z[kBond] = deformed_coordinates[node_j, 2] - deformed_coordinates[node_i, 2]

    deformed_length = numpy.sqrt(deformed_X ** 2 + deformed_Y ** 2 + deformed_Z ** 2)
    stretch = (deformed_length - bond_length) / bond_length

    return deformed_X, deformed_Y, deformed_Z, deformed_length, stretch


# --------------------------------------------
#      Calculate bond softening factor
# --------------------------------------------

# TODO: this function has not been tested

@njit
def calculate_bsf_trilinear(stretch, s0, s1, sc, bond_softening_factor, flag_bsf):

    nbonds = len(stretch)
    beta = 0.25
    eta = s1 / s0
    bsf = numpy.zeros(nbonds)

    for kBond in range(nbonds):

        if (stretch[kBond] > s0) and (stretch[kBond] <= s1):

            flag_bsf[kBond] = 1
            bsf[kBond] = 1 - ((eta - beta) / (eta - 1) * (s0 / stretch[kBond])) + ((1 - beta) / (eta - 1))

        elif (stretch[kBond] > s1) and (stretch[kBond] <= sc):

            bsf[kBond] = 1 - ((s0 * beta) / stretch[kBond]) * ((sc - stretch[kBond]) / (sc - s1))

        elif stretch[kBond] > sc:

            bsf[kBond] = 1

        if bsf[kBond] > bond_softening_factor[kBond]:  # Bond softening factor can only increase (damage is irreversible)

            bond_softening_factor[kBond] = bsf[kBond]

    return bond_softening_factor, flag_bsf


@njit
def calculate_bsf_non_linear(stretch, s0, sc, bond_softening_factor, flag_bsf):

    nbonds = len(stretch)
    k = 25
    alpha = 0.25
    denominator = 1 - numpy.exp(-k)
    bsf = numpy.zeros(nbonds)

    for kBond in range(nbonds):

        if (stretch[kBond] > s0) and (stretch[kBond] < sc):

            numerator = 1 - numpy.exp(-k * (stretch[kBond] - s0) / (sc - s0))
            residual = alpha * (1 - (stretch[kBond] - s0) / (sc - s0))
            bsf[kBond] = 1 - (s0 / stretch[kBond]) * ((1 - (numerator / denominator)) + residual) / (1 + alpha)

            flag_bsf[kBond] = 1

        elif stretch[kBond] > sc:

            bsf[kBond] = 1

        if bsf[kBond] > bond_softening_factor[kBond]:  # Bond softening factor can only increase (damage is irreversible)

            bond_softening_factor[kBond] = bsf[kBond]

    return bond_softening_factor, flag_bsf


# --------------------------------------------
#           Calculate bond force
# --------------------------------------------

@njit(nogil=True, parallel=True)
def calculate_bond_force(bond_stiffness, bond_softening_factor, stretch, cell_volume,
                         deformed_X, deformed_Y, deformed_Z, deformed_length):
    bond_force_X = bond_stiffness * (1 - bond_softening_factor) * stretch * cell_volume * (deformed_X / deformed_length)
    bond_force_Y = bond_stiffness * (1 - bond_softening_factor) * stretch * cell_volume * (deformed_Y / deformed_length)
    bond_force_Z = bond_stiffness * (1 - bond_softening_factor) * stretch * cell_volume * (deformed_Z / deformed_length)
    return bond_force_X, bond_force_Y, bond_force_Z


# --------------------------------------------
#           Calculate nodal force
# --------------------------------------------

@njit
def calculate_nodal_force(nnodes, bondlist, bond_force_X, bond_force_Y, bond_force_Z):

    nodal_force = numpy.zeros((nnodes, 3), dtype=numpy.float64)

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

# --------------------------------------------
#            Time integration
# --------------------------------------------

# @njit(nogil=True, parallel=True)
def euler_cromer(nodal_force, nodal_displacement, nodal_velocity, density, bc_type, bc_values, bc_scale, DT):
    damping = 0
    nodal_acceleration = (nodal_force - damping * nodal_velocity) / density
    nodal_acceleration[bc_type == 1] = 0  # Apply boundary conditions - constraints
    nodal_velocity_forward = nodal_velocity + (nodal_acceleration * DT)
    nodal_displacement_DT = nodal_velocity_forward * DT
    nodal_displacement_forward = nodal_displacement + nodal_displacement_DT
    nodal_displacement_forward[bc_values != 0] = bc_scale * -1  # TODO: fix this line
    return nodal_displacement_forward


# --------------------------------------------
#               Damage
# --------------------------------------------

@njit
def calculate_damage(n_family_members, bondlist, fail):

    nnodes = len(n_family_members)
    unbroken_bonds = numpy.zeros(nnodes)

    for kBond, bond in enumerate(bondlist):

        node_i = bond[0]
        node_j = bond[1]

        unbroken_bonds[node_i] = unbroken_bonds[node_i] + fail[kBond]
        unbroken_bonds[node_j] = unbroken_bonds[node_j] + fail[kBond]

    damage = 1 - (unbroken_bonds / n_family_members)

    return damage







