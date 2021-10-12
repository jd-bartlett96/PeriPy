# ----------------------------------------------
#      Beam 4 (175 mm x 50 mm x 50 mm)
# ----------------------------------------------

import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peripy import Model
from peripy.integrators import VelocityVerletCL
from peripy.utilities import write_array
from pstats import SortKey, Stats
from bc_utilities import (is_tip_5mm,
                          is_bond_type_5mm,
                          is_displacement_boundary_5mm,
                          smooth_step_data)  # Note: MH wrote smooth_step_data
import h5py
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["font.family"] = "Times New Roman"

os.environ['PYOPENCL_CTX'] = '0:0'


def is_density(x):
    """Determine the density of the node."""
    density_concrete = 2346.0
    return density_concrete

# ----------------------------------------------
#                 WARNING!
# ----------------------------------------------
# Surface corrections factors (to correct the peridynamic surface effect) and
# partial volume corrections factors (to improve spatial integration accuracy)
# are missing from this example


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_file_name",
                        help="run example on a given mesh file name")
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    mesh_file = pathlib.Path(__file__).parent.resolve() / args.mesh_file_name
    write_path_solutions = (pathlib.Path(__file__).parent.resolve() /
                            args.mesh_file_name.replace('.msh', ''))

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    # --------------------------------
    #           Constants
    # --------------------------------

    nnodes = 3620  # TODO: this should not be entered manually
    dx = 5.0e-3
    horizon = dx * np.pi
    s_0 = 1.05e-4
    s_1 = 6.90e-4
    s_c = 5.56e-3
    beta = 0.25
    c = 2.32e18
    # cell_volume = dx ** 3
    c_1 = (beta * c * s_0 - c * s_0) / (s_1 - s_0)
    c_2 = (- beta * c * s_0) / (s_c - s_1)
    critical_stretch_ = [np.float64(s_0), np.float64(s_1), np.float64(s_c)]
    critical_stretch_nf = [1000. * np.float64(s_0),
                           1000. * np.float64(s_1),
                           1000. * np.float64(s_c)]
    bond_stiffness_ = [np.float64(c), np.float64(c_1), np.float64(c_2)]
    bond_stiffness_nf = [np.float64(c), np.float64(c), np.float64(c)]
    bond_stiffness = [bond_stiffness_, bond_stiffness_nf]
    critical_stretch = [critical_stretch_, critical_stretch_nf]
    damping = 2.5e6
    saf_fac = 0.25
    dt = 1.3e-6
    steps = 100000  # Number of time steps
    applied_displacement = 2e-4
    volume = np.power(dx, 3) * np.ones(nnodes, dtype=np.float64)
    print('dt =', "{:.3e}".format(dt),
          '; safety_fac =', saf_fac,
          '; damping =', "{:.2e}".format(damping))

    # --------------------------------
    #       Boundary conditions
    # --------------------------------

    displacement_bc_array = smooth_step_data(0, steps, 0, applied_displacement)

    # --------------------------------
    #       Solver (integrator)
    # --------------------------------

    integrator = VelocityVerletCL(dt=dt, damping=damping)
    # integrator = EulerNumba(dt=dt, s0=s_0, s1=s_1, sc=s_c, c=c,
    #                         cell_volume=cell_volume)

    # --------------------------------
    #          Input file
    # --------------------------------

    model = Model(
        mesh_file, integrator=integrator, horizon=horizon,
        critical_stretch=critical_stretch, bond_stiffness=bond_stiffness,
        dimensions=3,
        is_density=is_density,
        volume=volume,
        is_bond_type=is_bond_type_5mm,
        is_displacement_boundary=is_displacement_boundary_5mm,
        is_tip=is_tip_5mm)

    # --------------------------------
    #          Simulation
    # --------------------------------

    write = 100  # write output data every 1000 time steps

    # TODO: (u, coords, damage, connectivity, f, ud, data) expected 7, got 6
    (u, damage, connectivity, f, ud, data) = model.simulate(
        bond_stiffness=bond_stiffness,
        critical_stretch=critical_stretch,
        steps=steps,
        displacement_bc_magnitudes=displacement_bc_array,
        write=write)

    # --------------------------------
    #          Output data
    # --------------------------------

    first = 0
    last = -1

    force = np.array(data['force']['body_force'][first:last]) / 1000
    left_displacement = 1000. * np.array(data['CMOD_left']['displacement']
                                         [first:last])
    right_displacement = 1000. * np.array(data['CMOD_right']['displacement']
                                          [first:last])
    CMOD = np.subtract(right_displacement, left_displacement)

    # Write load-CMOD data to disk
    try:
        write_array(write_path_solutions / "data.h5", "force", np.array(force))
        write_array(write_path_solutions / "data.h5", "CMOD", np.array(CMOD))
    except ValueError:  # data.h5 already exists
        os.remove(write_path_solutions / "data.h5")
        write_array(write_path_solutions / "data.h5", "force", np.array(force))
        write_array(write_path_solutions / "data.h5", "CMOD", np.array(CMOD))

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())

    # --------------------------------
    #         Post-processing
    # --------------------------------

    # Experimental data
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

    # Numerical data
    plt.plot(CMOD, -force, label='Numerical')
    plt.xlabel('CMOD [mm]')
    plt.ylabel('Force [kN]')
    plt.grid(True)
    axes = plt.gca()
    axes.set_xlim([0, .3])
    axes.tick_params(direction='in')

    # Verification data - https://github.com/mhobbs18/BB_PD
    ver_data_path = (pathlib.Path(__file__).parent.resolve()
                     / "verification_data.h5")
    ver_data = h5py.File(ver_data_path, 'r')
    ver_load_CMOD = ver_data['load_CMOD']
    ver_load = ver_load_CMOD[0, 0:499] / 1000
    ver_CMOD = ver_load_CMOD[1, 0:499]
    plt.plot(ver_CMOD, ver_load, 'tab:orange', label='Verification data')

    plt.legend()
    plt.savefig('load_CMOD', dpi=1000)


if __name__ == "__main__":
    main()
