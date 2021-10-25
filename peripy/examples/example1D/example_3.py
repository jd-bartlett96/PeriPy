"""
A very simple 1D problem used to test linear implementation of the peripy code.

"""

import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peripy.model_1D import Model_1D
from peripy import integrators
#from peripy.model import initial_crack_helper
from peripy.integrators import EulerCL, Euler, Implicit
from pstats import SortKey, Stats
from matplotlib import pyplot as plt


mesh_def = np.linspace(start=0, stop=1, num=10)
total_volume = max(mesh_def)


def is_displacement_boundary(x):
    """
    Returns a boolean for the displacement boundary along length of rod.
    Returns:
        None where there is no boundary condition;
        -1 where the boundary is displacement loaded in positive direction;
        1 where the boundary is displacement loaded in positive direction;
        0 where the boundary is clamped
    :arg x: Particle x coord along length of rod.
    :type x: :class:`numpy.float64`
    """

    if x < 0.1:
        bnd = 0
    elif x > 0.9:
        bnd = 0
    else:
        bnd = None
    return bnd

def is_force_boundary(x):

    """
    Return a boolean force boundary.

    Returns a boolean list, whose elements are:
        None where there is no boundary condition;
        -1 where the boundary is displacement loaded in negative direction;
        1 where the boundary is displacement loaded in positive direction;
        0 where the boundary is clamped;

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`
    """

    bnd = None
    if x < 0.9 and x > 0.1:
        bnd = np.random.normal(0, 1)

    return bnd

def main():
    "Conducts a peridynamics simulation"
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()


    # The :class:`peripy.integrators.Euler` class is the cython
    # implementation of the explicit Euler integration scheme.
    integrator = Implicit(dt=1e-3)

    # The bond_stiffness, also known as the micromodulus, of the peridynamic
    # bond, using Silling's (2005) derivation for the prototype microelastic
    # brittle (PMB) material model.
    # An arbritrary value of the critical_stretch = 0.005m is used.
    horizon = 0.125
    bond_stiffness = 18.00 * 0.05 / (np.pi * horizon**4)
    # The :class:`peripy.model.Model` defines and calculates the
    # connectivity of the model, as well as the boundary conditions and crack.
    model = Model_1D(
        mesh_def, integrator=integrator, horizon=horizon,
        critical_stretch=0.005, bond_stiffness=bond_stiffness,
        is_displacement_boundary=is_displacement_boundary, is_force_boundary=is_force_boundary,
        dimensions=2, initial_crack=None, volume_total=total_volume)

    # The simulation will have 1000 time steps, and last
    # dt * steps = 1e-3 * 1000 = 1.0 seconds
    steps = 1

    # The boundary condition magnitudes will be applied at a rate of
    # 2.5e-6 m per time-step, giving a total final displacement (the sum of the
    # left and right hand side) of 5mm.
    displacement_bc_magnitude = np.array([0.0005])
    force_bc_magnitudes = np.array([.001])


    # The :meth:`Model.simulate` method can be used to conduct a peridynamics
    # simulation. Here it is possible to define the boundary condition
    # magnitude throughout the simulation.
    u, damage, *_ = model.simulate(
        steps=steps,
        displacement_bc_magnitudes=displacement_bc_magnitude,
        force_bc_magnitudes=force_bc_magnitudes,
        write=1)
   

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    
    plt.scatter(mesh_def, u)
    plt.show()
    


if __name__ == "__main__":
    main()
 