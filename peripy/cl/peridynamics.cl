#pragma OPENCL EXTENSION cl_khr_fp64 : enable


#ifdef BOND_DAMAGE_PMB

void bond_damage_PMB(
    double stretch,
    double critical_stretch,
    __global int* nlist,
    int const global_id
)
{
    if (stretch < critical_stretch) {
        double damage = 0.0;
    }
    else {
        // break bond
        nlist[global_id] = -1;
        double damage = 1.0;
    }
    return damage
}

#elif BOND_DAMAGE_BILINEAR

void bond_damage_bilinear(
    double stretch,
    double critical_stretch,
    __global int* nlist,
    int const global_id
)
{
    if (stretch <= critical_stretch[0]){
        double damage = 0.0;
    }
    if (stretch > critical_stretch[0]) and (stretch <= critical_stretch[1]) {
        double eta = critical_stretch[0] / critical_stretch[1];
        double damage = 1 - (eta / (eta - 1) * (critical_stretch[0] / stretch_bond)) + (1 / (eta - 1));
    }
    elif (stretch > critical_stretch[1]){
        double damage = 1.0;
    }
    return damage
}

#elif BOND_DAMAGE_TRILINEAR

void bond_damage_trilinear(
    double stretch,
    double critical_stretch,
    __global int* nlist,
    int const global_id
)
{
    if (stretch <= critical_stretch[0]){
        double damage = 0.0;
    }
    if (stretch > critical_stretch[0]) and (stretch <= critical_stretch[1]) {
        double eta = critical_stretch[0] / critical_stretch[1];
        double damage = 1 - ((eta - beta) / (eta - 1) * (critical_stretch[0] / stretch_bond)) + ((1 - beta) / (eta - 1));
    }
    elif (stretch > critical_stretch[1]) and (stretch <= critical_stretch[2]) {
        double damage = 1 - (critical_stretch[0] * beta / stretch) * ((critical_stretch[2] - stretch_bond) / (critical_stretch[2] - critical_stretch[1]));
    }
    elif (stretch > critical_stretch[2]){
        double damage = 1.0;
    }
    return damage
}

#elif BOND_DAMAGE_EXPONENTIAL

void bond_damage_exponential(
    double stretch,
    double critical_stretch
)
{
    if (stretch <= critical_stretch[0]){
        double damage = 0.0;
    }
    if (stretch > critical_stretch[0]) and (stretch <= critical_stretch[1]) {
        double eta = critical_stretch[0] / critical_stretch[1];
        double damage = 1 - ((eta - beta) / (eta - 1) * (critical_stretch[0] / stretch_bond)) + ((1 - beta) / (eta - 1));
    }
    elif (stretch > critical_stretch[1]){
        double damage = 1.0;
    }
    return damage
}


#elif BOND_DAMAGE_SIGMOID

void bond_damage_sigmoid(
    double stretch,
    double critical_stretch,
    double 
)
{
    double damage = 1 / (exp(stretch - critical_stretch[0]) / critical_stretch[1] + 1);
    return damage
}

#endif


__kernel void
	bond_force(
    __global double const* u,
    __global double* force,
    __global double* body_force,
    __global double const* r0,
    __global double const* vols,
	__global int* nlist,
    __global int const* fc_types,
    __global double const* fc_values,
    __global double const* stiffness_corrections,
    __global int const* bond_types,
    __global double* bond_damage,
    __global double const* plus_cs,
    __local double* local_cache_x,
    __local double* local_cache_y,
    __local double* local_cache_z,
    double bond_stiffness,
    double critical_stretch,
    double fc_scale
	) {
    /* Calculate the force due to bonds on each node.
     *
     * This bond_force function is for the simple case of no stiffness corrections and no bond types.
     *
     * u - An (n,3) array of the current displacements of the particles.
     * force - An (n,3) array of the current forces on the particles.
     * body_force - An (n,3) array of the current internal body forces of the particles.
     * r0 - An (n,3) array of the coordinates of the nodes in the initial state.
     * vols - the volumes of each of the nodes.
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * fc_types - An (n,3) array of force boundary condition types,
     *     a value of 0 denotes a particle that is not externally loaded.
     * fc_values - An (n,3) array of the force boundary condition values applied to particles.
     * stiffness_corrections - Not applied in this bond_force kernel. Placeholder argument.
     * bond_types - Not applied in this bond_force kernel. Placeholder argument.
     * regimes - Not applied in this bond_force kernel. Placeholder argument.
     * plus_cs - Not applied in this bond_force kernel. Placeholder argument.
     * local_cache_x - local (local_size) array to store the x components of the bond forces.
     * local_cache_y - local (local_size) array to store the y components of the bond forces.
     * local_cache_z - local (local_size) array to store the z components of the bond forces.
     * bond_stiffness - The bond stiffness.
     * critical_stretch - The critical stretch, at and above which bonds will be broken.
     * fc_scale - scale factor appied to the force bondary conditions.
     * nregimes - Not applied in this bond_force kernel. Placeholder argument. */
    // global_id is the bond number
    const int global_id = get_global_id(0);
    // local_id is the LOCAL node id in range [0, max_neigh] of a node in this parent node's family
	const int local_id = get_local_id(0);
    // local_size is the max_neigh, usually 128 or 256 depending on the problem
    const int local_size = get_local_size(0);
	// group_id is the node i
	const int node_id_i = get_group_id(0);

	// Access local node within node_id_i's horizon with corresponding node_id_j,
	const int node_id_j = nlist[global_id];

	// If bond is not broken
	if (node_id_j != -1) {
		const double xi_x = r0[3 * node_id_j + 0] - r0[3 * node_id_i + 0];
		const double xi_y = r0[3 * node_id_j + 1] - r0[3 * node_id_i + 1];
		const double xi_z = r0[3 * node_id_j + 2] - r0[3 * node_id_i + 2];

		const double xi_eta_x = u[3 * node_id_j + 0] - u[3 * node_id_i + 0] + xi_x;
		const double xi_eta_y = u[3 * node_id_j + 1] - u[3 * node_id_i + 1] + xi_y;
		const double xi_eta_z = u[3 * node_id_j + 2] - u[3 * node_id_i + 2] + xi_z;

		const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
		const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
		const double stretch = (y -  xi)/ xi;

        // Apply the damage law
        // Note that ifdef checks for macro's existence, whereas elif checks for macro's value
        #ifdef BOND_DAMAGE_PMB
            bond_damage_PMB(bond_damage, nlist, stretch, critical_stretch);
        #elif BOND_DAMAGE_BILINEAR
            bond_damage_bilinear(bond_damage, nlist, stretch, critical_stretch, plus_cs);
        #elif BOND_DAMAGE_TRILINEAR
            bond_damage_trilinear(bond_damage, nlist, stretch, critical_stretch, plus_cs);
        #elif BOND_DAMAGE_EXPONENTIAL
            bond_damage_exponential(bond_damage, nlist, stretch, critical_stretch, plus_cs);
        #elif BOND_DAMAGE_SIGMOID
            bond_damage_sigmoid(bond_damage, stretch, critical_stretch);
        #endif

        const double cx = xi_eta_x / y;
		const double cy = xi_eta_y / y;
		const double cz = xi_eta_z / y;

        #ifdef STIFFNESS_CORRECTIONS
            const double f = stretch * bond_damage[global_id] * bond_stiffness * stiffness_corrections[global_id] * vols[node_id_j];
        #else
            const double f = stretch * bond_damage[global_id] * bond_stiffness * vols[node_id_j];
        #endif

        // Copy bond forces into local memory
		local_cache_x[local_id] = f * cx;
	    local_cache_y[local_id] = f * cy;
		local_cache_z[local_id] = f * cz;
    }
    // bond is broken
    else {
        local_cache_x[local_id] = 0.00;
        local_cache_y[local_id] = 0.00;
        local_cache_z[local_id] = 0.00;
    }

    // Wait for all threads to catch up
    barrier(CLK_LOCAL_MEM_FENCE);
    // Parallel reduction of the bond force onto node force
    for (int i = local_size/2; i > 0; i /= 2) {
        if(local_id < i) {
            local_cache_x[local_id] += local_cache_x[local_id + i];
            local_cache_y[local_id] += local_cache_y[local_id + i];
            local_cache_z[local_id] += local_cache_z[local_id + i];
        } 
        //Wait for all threads to catch up 
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!local_id) {
        //Get the reduced forces
        double const force_x = local_cache_x[0];
        double const force_y = local_cache_y[0];
        double const force_z = local_cache_z[0];
        // Update body forces in each direction
        body_force[3 * node_id_i + 0] = force_x;
        body_force[3 * node_id_i + 1] = force_y;
        body_force[3 * node_id_i + 2] = force_z;
        // Update forces in each direction
        force[3 * node_id_i + 0] = (fc_types[3 * node_id_i + 0] == 0 ? force_x : (force_x + fc_scale * fc_values[3 * node_id_i + 0]));
        force[3 * node_id_i + 1] = (fc_types[3 * node_id_i + 1] == 0 ? force_y : (force_y + fc_scale * fc_values[3 * node_id_i + 1]));
        force[3 * node_id_i + 2] = (fc_types[3 * node_id_i + 2] == 0 ? force_z : (force_z + fc_scale * fc_values[3 * node_id_i + 2]));
    }
}


__kernel void damage(
        __global double const *bond_damage,
        __global int const *family,
        __global double *damage,
        __local double *local_cache
    )
{
    /* Calculate the damage of each node.
     *
     * nlist - An (n, local_size) array containing the neighbour lists,
     *     a value of -1 corresponds to a broken bond.
     * family - An (n) array of the initial number of neighbours for each node.
     * n_neigh - An (n) array of the number of neighbours (particles bound) for
     *     each node.
     * damage - An (n) array of the damage for each node. 
     * local_cache - local (local_size) array to store the bond breakages.*/
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size

    //Copy values into local memory
    #ifdef PLASTIC_DAMAGE
    local_cache[local_id] = bond_damage[global_id];
    #else
    local_cache[local_id] = bond_damage[global_id] == 0.0 ? 0.0 : 1.0;
    #endif

    //Wait for all threads to catch up
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = local_size/2; i > 0; i /= 2){
        if(local_id < i){
            local_cache[local_id] += local_cache[local_id + i];
        }
        // Wait for all threads to catch up
        barrier(CLK_LOCAL_MEM_FENCE)
    }
    if (!local_id){
        // Get the reduced damages
        int node_id_i = get_group_id(0);
        // Update damage
        damage[node_id_i] = 1.0 - (double) local_cache[0] / (double) family[node_id_i];
    }
}
