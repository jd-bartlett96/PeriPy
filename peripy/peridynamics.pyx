from .spatial cimport ceuclid, cstrain, cstrain2
import numpy as np


def damage(int[:] n_neigh, int[:] family):
    """
    Calculate the damage for each node.

    Damage is defined as the ratio of broken bonds at a node to the total
    number of bonds at that node.

    :arg n_neigh: The current number of neighbours for each node, *i.e.* the
        number of unbroken bonds. dtype=numpy.int32.
    :type n_neigh: :class:`numpy.ndarray`
    :arg family: The total (initial) number of bonds for each node.
        dtype=numpy.int32.
    :type family: :class:`numpy.ndarray`
    """
    cdef nnodes = family.shape[0]

    result = np.empty(nnodes, dtype=np.float64)
    cdef double[:] result_view = result

    cdef double ifamily
    cdef int i

    for i in range(nnodes):
        ifamily = family[i]
        result_view[i] = (ifamily - n_neigh[i])/ifamily

    return result

def bond_force(double[:, :] r, double[:, :] r0, int[:, :] nlist,
               int[:] n_neigh, double[:] volume, double bond_stiffness,
               double[:, :] force_bc_values, int[:, :] force_bc_types,
               double force_bc_scale):
    """
    Calculate the force due to bonds on each node.
    :arg r: The current coordinates of each node.
    :type r: :class:`numpy.ndarray`
    :arg r0: The initial coordinates of each node.
    :type r0: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg volume: The volume of each node.
    :type volume: :class:`numpy.ndarray`
    :arg float bond_stiffness: The bond stiffness.
    :arg force_bc_values: The force boundary condition values for each node.
    :type force_bc_values: :class:`numpy.ndarray`
    :arg force_bc_types: The force boundary condition types for each node.
    :type force_bc_types: :class:`numpy.ndarray`
    :arg double bc_scale: The scalar value applied to the
        force boundary conditions.
    """
    cdef int nnodes = nlist.shape[0]

    force = np.zeros((nnodes, 1), dtype=np.float64)
    cdef double[:, :] force_view = force

    cdef int i, j, dim, i_n_neigh, neigh
    cdef double strain, l, force_norm, nu, partial_volume_i, partial_volume_j
    cdef double[1] f

    for i in range(nnodes):
        i_n_neigh = n_neigh[i]
        for neigh in range(i_n_neigh):
            j = nlist[i, neigh]

            if i < j:
                # Calculate total force
                l = ceuclid(r[i], r[j])
                strain = cstrain2(l, r0[i], r0[j])
                force_norm = strain * bond_stiffness

                # Calculate component of force in each dimension
                force_norm = force_norm / l
                for dim in range(1):
                    f[dim] = force_norm * (r[j, dim] - r[i, dim])

                # Add force to particle i, using Newton's third law subtract
                # force from j
                # Scale the force by the partial volume of the child particle
                for dim in range(1):
                    force_view[i, dim] = (force_view[i, dim]
                                          + f[dim] * volume[j])
                    force_view[j, dim] = (force_view[j, dim]
                                          - f[dim] * volume[i])

        # Apply boundary conditions
        for dim in range(1):
            if force_bc_types[i, dim] != 0:
                force_view[i, dim] = force_view[i, dim] + (
                    force_bc_scale * force_bc_values[i, dim])

    return force


def break_bonds(double[:, :] r, double[:, :] r0, int[:, :] nlist,
                int[:] n_neigh, double critical_strain):
    """
    Update the neighbour list and number of neighbours by breaking bonds which
    have exceeded the critical strain.
    :arg r: The current coordinates of each node.
    :type r: :class:`numpy.ndarray`
    :arg r0: The initial coordinates of each node.
    :type r0: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg float critical_strain: The critical strain.
    """
    cdef int nnodes = nlist.shape[0]

    cdef int i, j, i_n_neigh, neigh
    cdef int j_n_neigh, jneigh

    # Check neighbours for each node
    for i in range(nnodes):
        # Get current number of neighbours
        i_n_neigh = n_neigh[i]

        neigh = 0
        while neigh < i_n_neigh:
            j = nlist[i, neigh]

            if i < j:
                if abs(cstrain(r[i], r[j], r0[i], r0[j])) < critical_strain:
                    # Move onto the next neighbour
                    neigh += 1
                else:
                    # Remove this neighbour by replacing it with the last
                    # neighbour on the list, then reducing the number of
                    # neighbours by 1.
                    # As neighbour `neigh` is now a new neighbour, we do not
                    # advance the neighbour index
                    nlist[i, neigh] = nlist[i, i_n_neigh-1]
                    i_n_neigh -= 1

                    # Remove from j
                    j_n_neigh = n_neigh[j]
                    for jneigh in range(j_n_neigh):
                        if nlist[j, jneigh] == i:
                            nlist[j, jneigh] = nlist[j, j_n_neigh-1]
                            n_neigh[j] = n_neigh[j] - 1
                            break
            else:
                # Move onto the next neighbour
                neigh += 1

        n_neigh[i] = i_n_neigh


def update_displacement(double[:, :] u, double[:, :] bc_values, 
                        int[:, :] bc_types, double[:, :] force, 
                        double bc_scale, double dt):
    """
    Update the displacement of each node for each node using an Euler
    integrator.
    :arg u: The current displacements of each node.
    :type u: :class:`numpy.ndarray`
    :arg bc_values: An (n,3) array of the boundary condition values.
    :type bc_values: :class:`numpy.ndarray`
    :arg bc_types: An (n,3) array of the boundary condition types, where a
        zero value represents an unconstrained node.
    :type bc_types: :class:`numpy.ndarray`
    :arg force: The force due to bonds on each node.
    :type force: :class:`numpy.ndarray`
    :arg float bc_scale: The scalar value applied to the
        displacement boundary conditions.
    :arg float dt: The length of the timestep in seconds.
    """
    # TO DO: changed the for loop to stop at 1 not 3.
    # Must check that this is correct. I.e. is it looping through
    # Num of dimensions or the num boundary coniditons?

    cdef int nnodes = u.shape[0]
    for i in range(nnodes):
            for dim in range(1):
                if bc_types[i, dim] == 0:
                    u[i, dim] = u[i, dim] + dt * force[i, dim]
                else:
                    u[i, dim] = bc_scale * bc_values[i, dim]

def assemble_K_global(double[:, :] r, int[:, :] nlist, int[:] n_neigh, double[:] volume,  
                    double bond_stiffness, double[:] bc_values, double[:] bc_types):

    """
    Calculate the bond stiffnesses for each node pairing and
    assemble into a global K matrix.
    :arg r: The current coordinates of each node.
    :type r: :class:`numpy.ndarray`
    :arg r0: The initial coordinates of each node.
    :type r0: :class:`numpy.ndarray`
    :arg nlist: The neighbour list
    :type nlist: :class:`numpy.ndarray`
    :arg n_neigh: The number of neighbours for each node.
    :type n_neigh: :class:`numpy.ndarray`
    :arg volume: The volume of each node.
    :type volume: :class:`numpy.ndarray`
    :arg float bond_stiffness: The bond stiffness.
    :arg force_bc_values: The force boundary condition values for each node.
    :type force_bc_values: :class:`numpy.ndarray`
    :arg force_bc_types: The force boundary condition types for each node.
    :type force_bc_types: :class:`numpy.ndarray`
    :arg double bc_scale: The scalar value applied to the
        force boundary conditions.    
    """

    cdef int nnodes = nlist.shape[0]
    cdef int i, j, dim, i_n_neigh, neigh, v, w
    cdef double strain, l, nu, partial_volume_i, partial_volume_j, multiplier
    cdef double[:, :] K_local, C

    # Loop through each of the nodes and work out their bond stiffnesses
    # Assemble and reduce to get K_tangent.

    K_global = np.zeros((nnodes, nnodes), dtype=np.float64)
    cdef double[:, :] K_matrix = K_global

    for i in range(nnodes):
        i_n_neigh = n_neigh[i]  # find number of neighbours for current node
        for neigh in range(i_n_neigh):  # loop over all neighbours of ith node
            j = nlist[i, neigh]

            if i < j and j != -1:   # only need to loop over first half since, others will be covered by N3L.
                l = ceuclid(r[i], r[j])     # find separation
                multiplier = ((bond_stiffness/(l)) * (volume[i]*volume[j])) #TO DO: need volume correction factor and softening factor here
                # Build the local stiffness matrix (only works for 1D):
                K_local = np.array([
                    [multiplier, -(multiplier)],
                    [-(multiplier), multiplier]
                ])
                
                # There is a better way to do this by saving the indices but this works.                
                K_global[i][i] += K_local[0][0]
                K_global[i][j] += K_local[0][1]
                K_global[j][i] += K_local[1][0]
                K_global[j][j] += K_local[1][1]
    

    return K_global


def find_displacements_implicit(double[:, :] K_global, double[:, :] r, double displacement_bc_magnitude,
                        double[:] bc_types, double[:] bc_values):

    """
    Takes a preevaluated global stiffness matrix and displacement boundary
    conditions and uses a constraint matrix to create a reduced stiffness 
    matrix. Build the rhs vector and solves the system Ku = f for the 
    displacements.
    
    """

    cdef double[:] u_eff, u, unconstrained_dofs
    cdef double[:, :] C, u_return
    cdef int i,j, n_bc, nnodes, DOF 
    
    nnodes = r.shape[0]
    DOF = r.shape[1]

    n_bc = 0
    # Count number of constrained nodes.
    for entry in bc_types:
        if entry != 0:
            n_bc += 1

    i = 0
    j = 0
    # Build a constraint matrix. All rows with no BC form an I matrix,
    # those with a BC are zeroed.
    C = np.zeros((nnodes, nnodes - n_bc), dtype=np.float64)
    for i in range(nnodes):
        if bc_types[i] != 0:
            continue
        C[i][j] = 1
        j += 1

    # Build the reduced K matrix
    K_reduced = np.matmul(np.matmul(np.transpose(C), K_global), C)
    
    # Initiate vectors for displacements and flags for unconstrained DOFs
    # and store number of nodes.
    nnodes = r.shape[0]
    u = np.zeros(nnodes) * DOF
    unconstrained_dofs = np.zeros(nnodes - n_bc)

    # Assemble the known DOFs into a vector u. Create a vector of the indices 
    # of unconstrained DOFs in vector r --> length = nnodes - n_bc.
    j = 0
    for i in range(nnodes):
        if bc_types[i] != 0: 
            if bc_values[i] == 0:
                u[i] = r[i][0]
            elif bc_values[i] == 1:
                u[i] = r[i][0] + displacement_bc_magnitude
            elif bc_values[i] == -1:
                u[i] = r[i][0] - displacement_bc_magnitude
        else:
            unconstrained_dofs[j] = i
            j += 1

    # Transform u into u_eff - the reduction term which is taken 
    # off the unconstrained DOF's force values.
    u_eff = -1 * np.matmul(np.matmul(np.transpose(C), K_global), u)
    
    # Solve for the actual positions of the unconstrained nodes.
    x = np.linalg.solve(K_reduced, u_eff)

    # Now update u vector with the unconstrained DOF's found in x.
    i = 0
    for entry in unconstrained_dofs:
        entry = int(entry)
        u[entry] = x[i]
        i += 1
    # Subtract original positions to get displacements. Update positions.
    i = 0
    for i in range(nnodes):
        u[i] += -1 * r[i][0]
        r[i][0] += u[i]    

    return u