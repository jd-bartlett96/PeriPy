"""
A simple, 3D perdiynamics simulation example.

This example is compared to real life experimental data from Grégoire et al.
[1], and also verified against Mark Hobbs peridynamics code's [2] simulation
data, which was developed independently from PeriPy.

This example is a 0.175m x 0.05m x 0.05m plain concrete, simply supported
beam with no pre-crack, subjected to displacement controlled three-point
bending. The displacement loading follows a 5th order polynomial so that
acceleration of the boundaries is zero at the start and end of the simulation.
The span between the supports is 0.125m. The model uses a ``trilinear"
constitutive model. No surface correction factors or partial volume correction
factors are applied to the model.

In this example, the first time the volume, family and connectivity of the
model are calculated, they are also stored in file `175beam3620.h5'.
In subsequent simulations, the arrays are loaded from this h5 file instead of
being calculated again, therefore reducing the overhead of initiating the
model.

The output is a load-CMOD curve which is compared to the experimental data [1]
and to the verification data [2].

[1] Grégoire, D., Rojas-Solano, L. B., and Pijaudier-Cabot, G. (2013).
Failure and size effect for notched and unnotched concrete beams.
International Journal for Numerical and Analytical Methods in Geomechanics,
37(10):1434–1452.

[2] Mark Hobbs (2019), BB_PD, https://github.com/mhobbs18/BB_PD
"""
import os
import argparse
import cProfile
from io import StringIO
import pathlib

from pstats import SortKey, Stats
import numpy as np
import h5py
import matplotlib.pyplot as plt

from peripy import Model
from peripy.integrators import VelocityVerletCL, EulerNumba_nlist, EulerNumba_blist
from peripy.utilities import write_array
from peripy.utilities import read_array as read_model

# The .msh file is a point cloud. '175beam3620.msh' contains 3620 particles.
mesh_file = pathlib.Path(__file__).parent.resolve() / '175beam3620.msh'


def smooth_step_data(
        first_step, steps, start_value, final_value):
    """
    Apply a smooth 5th order polynomial as the displacement-time function for
    the simulation.

    :arg int start_time_step: The time step that the simulation starts from.
    :arg int steps: The number of steps in the simulation.
    :arg float start_value: The starting value of the displacement.
    :arg float final_value: The final value of the displacement.
    """
    alpha = np.zeros(first_step + steps)
    for current_time_step in range(first_step, first_step + steps):
        xi = (current_time_step - first_step) / steps
        alpha[current_time_step] = (start_value + (
            final_value - start_value) * xi**3 * (10 - 15 * xi + 6 * xi**2))
    return alpha


def is_tip(x):
    """
    Return a boolean list of tip types for each cartesian direction.

    Returns a boolean list, whose elements are a strings when the particle
    resides on a 'tip' to be measured for some displacement, velocity,
    acceleration, force or body_force in that cartesian direction. The value
    of the element of the list can be a string or an int, which is a flag for
    the tip type that the particle resides on. If a particle resides on more
    than one tip, then any of the list elements can be a tuple of tip types.

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`

    :returns: A (3,) list of tip types.
    :rtype: List of (tuples of) None or string.
    """
    # Particle does not live on tip
    tip = [None, None, None]
    # Particle does live on a tip to me measured, in the centre of the beam
    if (x[0] > 0.0875 - 5.0e-3) and (x[0] < 0.0875 + 5.0e-3):
        if (x[2] > 0.025 - 5.0e-3) and (x[2] < 0.025 + 5.0e-3):
            # Measurements are made in the z direction
            tip[2] = 'deflection'
    # Measure the force applied at the boundaries, in the z direction
    if x[0] == 0.025 and x[2] == -0.01:
        tip[2] = 'force'
    if x[0] == 0.155 and x[2] == -0.01:
        tip[2] = 'force'
    # Measurement of the "crack mouth opening displacement" (CMOD)
    if x[0] == 0.125 and x[1] == 0.005 and x[2] == 0.005:
        tip[0] = 'CMOD_right'
    if x[0] == 0.055 and x[1] == 0.005 and x[2] == 0.005:
        tip[0] = 'CMOD_left'
    return tip


def is_density(x):
    """
    Return the density of the particle.

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`

    :returns: density of concrete in [kg/m^3]
    :rtype: float
    """
    return 2346.0


def is_displacement_boundary(x):
    """
    Return a boolean list of displacement boundarys for each direction.

    Returns a (3,) boolean list, whose elements are:
        None where there is no boundary condition;
        -1 where the boundary is displacement loaded in negative direction;
        1 where the boundary is displacement loaded in positive direction;
        0 where the boundary is clamped;

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`
    """
    # Particle does not live on a boundary
    bnd = [None, None, None]
    # Displacement controlled particles
    if x[0] == 0.085 and x[2] == 0.05:
        bnd[2] = -1
    if x[0] == 0.09 and x[2] == 0.05:
        bnd[2] = -1
    if x[0] == 0.095 and x[2] == 0.05:
        bnd[2] = -1
    # Clamped particles
    if (x[0] > 0.025 - 5e-3) and (x[0] < 0.025 + 5e-3) and (x[2] < -0.005):
        bnd[1] = 0
        bnd[2] = 0
    if (x[0] > 0.155 - 5e-3) and (x[0] < 0.155 + 5e-3) and (x[2] < -0.005):
        bnd[1] = 0
        bnd[2] = 0
    return bnd


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    write_path_model = (pathlib.Path(__file__).parent.absolute() / str(
        '175beam3620_model.h5'))
    write_path_solutions = pathlib.Path(__file__).parent.absolute()

    # Constants
    # Average one-dimensional grid separation between particles along an axis
    dx = 5.0e-3
    # Following convention, the horizon distance is taken as just over 3
    # times the grid separation between particles
    horizon = dx * np.pi
    # Values of the particular trilinear constitutive model parameters
    s_0 = 1.05e-4
    s_1 = 6.90e-4
    s_c = 5.56e-3
    beta = 0.25
    c = 2.32e18
    c_1 = (beta * c * s_0 - c * s_0) / (s_1 - s_0)
    c_2 = (- beta * c * s_0) / (s_c - s_1)
    # Critical stretch for the trilinear constitutive model
    critical_stretch = [np.float64(s_0), np.float64(s_1), np.float64(s_c)]
    # Bond stiffness for the trilinear constitutive model
    bond_stiffness = [np.float64(c), np.float64(c_1), np.float64(c_2)]

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    # Setting the dynamic relaxation damping constant to a critical value
    # will help the system to converge to the quasi-static steady-state.
    damping = 2.5e6
    # Stable time step. Try increasing or decreasing it.
    dt = 1.3e-6  # 1.3e-15
    # integrator = VelocityVerletCL(dt=dt, damping=damping)
    integrator = EulerNumba_blist(dt=dt)

    # Try reading volume, density, family and connectivity arrays from
    # the file ./175beam3620_model.h5
    # volume is an (nnodes, ) :class:`np.ndarray` of particle volumes, where
    # nnodes is the number of particles in the .msh file
    volume = read_model(write_path_model, "volume")
    # density is an (nnodes, ) :class:`np.ndarray` of particle densities
    density = read_model(write_path_model, "density")
    # family is an (nnodes, ) :class:`np.ndarray` of initial number of
    # neighbours for each particle
    family = read_model(write_path_model, "family")
    # nlist is an (nnodes, max_neigh) :class:`np.ndarray` of the neighbours
    # for each particle. Each neigbour is given an integer i.d. in the range
    # [0, nnodes). max_neigh is atleast as large as np.max(family)
    nlist = read_model(write_path_model, "nlist")
    # n_neigh is an (nnodes, ) :class:`np.ndarray` of current number of
    # neighbours for each particle.
    n_neigh = read_model(write_path_model, "n_neigh")
    # The connectivity of the model is the tuple (nlist, n_neigh)
    if ((nlist is not None) and (n_neigh is not None)):
        connectivity = (nlist, n_neigh)
    else:
        connectivity = None

    if ((volume is not None) and
            (density is not None) and
            (family is not None) and
            (connectivity is not None)):
        # Model has been initiated before, so to avoid calculating volume,
        # family and connectivity arrays again, we can pass them as arguments
        # to the Model class
        model = Model(
            mesh_file, integrator=integrator, horizon=horizon,
            critical_stretch=critical_stretch, bond_stiffness=bond_stiffness,
            dimensions=3, family=family,
            volume=volume, connectivity=connectivity,
            density=density,
            is_displacement_boundary=is_displacement_boundary,
            is_tip=is_tip)
    else:
        # This is the first time that Model has been initiated, so the volume,
        # family and connectivity = (nlist, n_neigh) arrays will be calculated
        # and written to the file at location "write_path_model"
        model = Model(
            mesh_file, integrator=integrator, horizon=horizon,
            critical_stretch=critical_stretch, bond_stiffness=bond_stiffness,
            dimensions=3,
            volume_total=0.0004525,  # Total volume of the concrete beam
            is_density=is_density,
            is_displacement_boundary=is_displacement_boundary,
            is_tip=is_tip,
            write_path=write_path_model)

    # Visualise the particular constitutive model being used
    bond_stiffness, critical_stretch, plus_cs, *_ = model._set_damage_model(
        bond_stiffness, critical_stretch)
    plt.plot(
        [0, critical_stretch[0]],
        [0, critical_stretch[0] * bond_stiffness[0] + plus_cs[0]],
        color='k')
    plt.plot(
        [critical_stretch[0], critical_stretch[1]],
        [critical_stretch[0] * bond_stiffness[0],
            critical_stretch[1] * bond_stiffness[1] + plus_cs[1]],
        color='k')
    plt.plot(
        [critical_stretch[1], critical_stretch[2]],
        [critical_stretch[1] * bond_stiffness[1] + plus_cs[1],
            critical_stretch[2] * bond_stiffness[2] + plus_cs[2]],
        color='k')
    plt.xlabel('bond stretch (dimensionless)')
    plt.ylabel('bond force [N/m^6]')
    plt.title('Trilinear constitutive model')
    plt.savefig('trilinear_constitutive_model', dpi=1000)
    plt.close()

    # The displacement boundary condition magnitudes increase with a 5th order
    # polynomial so that the acceleration at the start and end of the
    # simulation is zero. The displacement of 0.2mm is applied over
    # 100,000 time-steps.
    steps = 100000
    applied_displacement = 2e-4
    displacement_bc_array = smooth_step_data(0, steps, 0, applied_displacement)

    # Run the simulation
    # Use e.g. paraview to view the output .vtk files of simulate
    (u, damage, connectivity, f, ud, data) = model.simulate(
        bond_stiffness=bond_stiffness,
        critical_stretch=critical_stretch,
        steps=steps,
        displacement_bc_magnitudes=displacement_bc_array,
        write_mesh=5000,  # write to mesh every 5000 time steps
        write_data=100)  # write to data every 100 time steps

    # TODO: why has this changed from data['force']['bodyforce'] to
    # data['force']['force']. Need to be more explicit with naming.
    # For example, nodal_force not force.
    # force = np.array(data['force']['body_force']) / 1000
    force = np.array(data['force']['force']) / 1000
    left_displacement = 1000. * np.array(data['CMOD_left']['displacement'])
    right_displacement = 1000. * np.array(data['CMOD_right']['displacement'])
    CMOD = np.subtract(right_displacement, left_displacement)

    # Write load-CMOD data to disk
    try:
        write_array(write_path_solutions / "data.h5", "force", np.array(force))
        write_array(write_path_solutions / "data.h5", "CMOD", np.array(CMOD))
    except OSError:  # data.h5 already exists
        os.remove(write_path_solutions / "data.h5")
        write_array(write_path_solutions / "data.h5", "force", np.array(force))
        write_array(write_path_solutions / "data.h5", "CMOD", np.array(CMOD))

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())

    # Plot the experimental data
    exp_data_path = (pathlib.Path(__file__).parent.resolve()
                     / "experimental_data.h5")
    exp_data = h5py.File(exp_data_path, 'r')
    exp_load_CMOD = exp_data['load_CMOD']
    exp_CMOD = exp_load_CMOD[0, 0:20000]
    exp_load_mean = exp_load_CMOD[1, 0:20000]
    exp_load_min = exp_load_CMOD[2, 0:20000]
    exp_load_max = exp_load_CMOD[3, 0:20000]
    plt.plot(exp_CMOD, exp_load_mean, color=(0.8, 0.8, 0.8),
             label='Experimental')
    plt.fill_between(exp_CMOD, exp_load_min, exp_load_max,
                     color=(0.8, 0.8, 0.8))

    # Plot the verification data
    ver_data_path = (pathlib.Path(__file__).parent.resolve()
                     / "verification_data.h5")
    ver_data = h5py.File(ver_data_path, 'r')
    ver_load_CMOD = ver_data['load_CMOD']
    ver_load = ver_load_CMOD[0, 0:499] / 1000
    ver_CMOD = ver_load_CMOD[1, 0:499]
    plt.plot(ver_CMOD, ver_load, 'tab:orange', label='Verification data')

    # Plot the numerical data
    force = read_model(write_path_solutions / "data.h5", "force")
    CMOD = read_model(write_path_solutions / "data.h5", "CMOD")
    plt.plot(CMOD, -force, label='Numerical')
    plt.xlabel('CMOD [mm]')
    plt.ylabel('Force [kN]')
    plt.grid(True)
    axes = plt.gca()
    axes.set_xlim([0, .3])
    axes.tick_params(direction='in')
    plt.legend()
    plt.savefig('load_CMOD', dpi=1000)
    plt.close()


if __name__ == "__main__":
    main()
