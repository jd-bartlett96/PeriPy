"""Utilities for example 3."""
import numpy as np

def is_displacement_boundary_5mm(x):
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
        bnd[2] = 0
        bnd[1] = 0

    if (x[0] > 0.155 - 5e-3) and (x[0] < 0.155 + 5e-3) and (x[2] < -0.005):
        bnd[2] = 0
        bnd[1] = 0

    return bnd


def is_tip_5mm(x):
    """
    Return a boolean list of tip types for each cartesian direction.

    Returns a boolean list, whose elements are True when the particle is to
    be measured for some displacement, velocity, force or acceleration
    in that cartesian direction.

    :arg x: Particle coordinate array of size (3,).
    :type x: :class:`numpy.ndarray`
    """
    # Particle does not live on tip
    tip = [None, None, None]


    if (x[0] > 0.0875 - 5.0e-3) and (x[0] < 0.0875 + 5.0e-3):
        if (x[2] > 0.025 - 5.0e-3) and (x[2] < 0.025 + 5.0e-3):
            tip[2] = 'deflection'


    if x[0] == 0.025 and x[2] == -0.01:
        tip[2] = 'force'

    if x[0] == 0.155 and x[2] == -0.01:
        tip[2] = 'force'


    if x[0] == 0.125 and x[1] == 0.005 and x[2] == 0.005:
        tip[0] = 'CMOD_right'

    if x[0] == 0.055 and x[1] == 0.005 and x[2] == 0.005:
        tip[0] = 'CMOD_left'

    return tip


def is_bond_type_5mm(x, y):
    """is_bond_type."""

    bnd = 0 # Bond can break

    return bnd

def _increment_displacement(coefficients, build_time, step, ease_off,
                            max_displacement_rate, build_displacement,
                            max_displacement):
    """
    Increment the displacement boundary condition values.

    According to a 5th order polynomial/ linear displacement-time curve
    for which initial acceleration is 0.

    :arg tuple coefficients: Tuple containing the 3 free coefficients
        of the 5th order polynomial.
    :arg int build_time: The number of time steps over which the
        applied displacement-time curve is not linear.
    :arg int step: The current time-step of the simulation.
    :arg int ease_off: A boolean-like variable which is 0 if the
        displacement-rate hasn't started decreasing yet. Equal to the step
        at which the displacement rate starts decreasing once it does so.
    :arg float max_displacement_rate: The displacement rate in [m] per step
        during the linear phase of the displacement-time graph.
    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg float max_displacement: The final applied displacement in [m].
    :returns: The displacement_bc_magnitude between [0.0, max_displacement],
        a scale applied to the displacement boundary conditions.
    :rtype: np.float64
    :returns: ease_off
    :rtype: int
    """
    if not ((max_displacement_rate is None) or (build_displacement is None)
            or (max_displacement is None)):
        # Calculate the scale applied to the displacements
        displacement_bc_magnitude, ease_off = calc_displacement_magnitude(
            coefficients, max_displacement, build_time,
            max_displacement_rate, step, build_displacement, ease_off)
    # No specified build up parameters
    elif max_displacement_rate is not None:
        # Increase displacement in linear increments
        displacement_bc_magnitude = max_displacement_rate * step
    elif max_displacement is not None:
        # Increase displacement in linear increments
        displacement_bc_magnitude = max_displacement / step
    return displacement_bc_magnitude, ease_off


def _increment_load(build_load_steps, max_load, step):
    """
    Increment and update the force boundary conditions.

    :arg float build_load_steps: The inverse of the number of steps
        required to build up to full external force loading.
    :arg float max_load: The maximum total external load in [N] applied to the
        loaded nodes.
    :arg int step: The current time-step of the simulation.
    :returns: The force_bc_magnitude between [0.0, max_load], in [N], a
        magnitude applied to the force boundary conditions.
    :rtype: :class:`numpy.float64`
    """
    # Increase load in linear increments
    if build_load_steps is not None:
        force_bc_magnitude = np.float64(
            min(1.0, build_load_steps * step) * max_load)
    else:
        force_bc_magnitude = np.float64(max_load)
    return force_bc_magnitude


def _calc_midpoint_gradient(T, displacement):
    """
    Calculate the midpoint gradient and coefficients of a 5th order polynomial.

    Calculates the midpoint gradient and coefficients of a 5th order
    polynomial displacement-time curve which is defined by acceleration being
    0 at t=0 and t=T and a total displacement.
    :arg int T: The total time in number of time steps of the smooth 5th order
        polynomial.
    :arg float displacement: The final displacement in [m] of the smooth 5th
        order polynomial.
    :returns: A tuple containing the midpoint gradient of the
        displacement-time curve and a tuple containing the 3 unconstrained
        coefficients of the 5th-order polynomial.
    :rtype: A tuple containing (:type float:, :type tuple:)
    """
    A = np.array([
        [1 * T**5, 1 * T**4, 1 * T**3],
        [20 * T**3, 12 * T**2, 6 * T],
        [5 * T**4, 4 * T**3, 3 * T**2]
        ]
        , dtype=np.float64)
    b = np.array(
        [
            [displacement],
            [0.0],
            [0.0]
                ], dtype=np.float64)
    x = np.linalg.solve(A, b)
    a = x[0][0]
    b = x[1][0]
    c = x[2][0]
    midpoint_gradient = (
        5 * a * (T / 2)**4 + 4 * b * (T/2)**3 + 3 * c * (T/2)**2)
    coefficients = (a, b, c)
    return(midpoint_gradient, coefficients)


def calc_displacement_magnitude(
        coefficients, max_displacement, build_time, max_displacement_rate,
        step, build_displacement, ease_off):
    """
    Calculate the displacement scale.

    Calculates the displacement boundary condition magnitude according to a
    5th order polynomial/ linear displacement-time curve for which initial
    acceleration is 0.

    :arg tuple coefficients: Tuple containing the 3 free coefficients
        of the 5th order polynomial.
    :arg float max_displacement: The final applied displacement in [m].
    :arg int build_time: The number of time steps over which the
        applied displacement-time curve is not linear.
    :arg float max_displacement_rate: The maximum displacement rate
        in [m] per step, which is the displacement rate during the linear phase
        of the displacement-time graph.
    :arg int step: The current time-step of the simulation.
    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg int ease_off: A boolean-like variable which is 0 if the
        displacement-rate hasn't started decreasing yet. Equal to the step
        at which the displacement rate starts decreasing once it does so.
    :returns: The displacement_bc_magnitude between
        [0.0, max_displacement], a scale applied to the displacement
        boundary conditions.
    :rtype: np.float64
    """
    a, b, c = coefficients
    # Acceleration part of displacement-time curve.
    if step < build_time / 2:
        displacement_bc_magnitude = a * step**5 + b * step**4 + c * step**3
    # Deceleration part of displacement-time curve.
    elif ease_off != 0:
        t = step - ease_off + build_time / 2
        if t > build_time:
            displacement_bc_magnitude = max_displacement
        else:
            displacement_bc_magnitude = (
                a * t**5 + b * t**4 + c * t**3
                + max_displacement
                - build_displacement)
    # Constant velocity
    else:
        # Calculate displacement.
        linear_time = step - build_time / 2
        linear_displacement = linear_time * max_displacement_rate
        displacement = linear_displacement + build_displacement / 2
        if displacement + build_displacement / 2 < max_displacement:
            displacement_bc_magnitude = displacement
        else:
            ease_off = step
            displacement_bc_magnitude = (
                max_displacement - build_displacement / 2)
    return(displacement_bc_magnitude, ease_off)


def calc_build_time(build_displacement, max_displacement_rate, steps):
    """
    Calculate the the number of steps for the 5th order polynomial.

    An iterative procedure to calculate the number of steps over which the
    displacement-time curve is a smooth 5th order polynomial.

    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg float max_displacement_rate: The displacement rate in [m] per step
            during the linear phase of the displacement-time graph.
    :arg int step: The current time-step of the simulation.
    :returns: A tuple containing an int T the number of steps over which the
        displacement-time curve is a smooth 5th order polynomial and a tuple
        containing the 3 unconstrained coefficients of the 5th-order
        polynomial.
    :rtype: A tuple containing (:type int:, :type tuple:)
    """
    build_time = 0
    midpoint_gradient = np.inf
    while midpoint_gradient > max_displacement_rate:
        # Try to calculate gradient
        try:
            midpoint_gradient, coefficients = _calc_midpoint_gradient(
                build_time, build_displacement)
        # No solution, so increase the build_time
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                pass
        build_time += 1
        if build_time > steps:
            raise ValueError(
                "Displacement build-up time was larger than total simulation "
                "time steps! \nTry increasing steps, decreasing "
                "build_displacement, or increasing max_displacement_rate. "
                "steps = {}".format(steps))
            break
    return(build_time, coefficients)


def calc_boundary_conditions_magnitudes(
        steps, max_displacement_rate, max_displacement=None,
        build_displacement=0, build_load_steps=None, max_load=None):
    """
    Calculate the boundary condition magnitudes arrays.

    Calculates the magnitude applied to the boundary conditions for each time-
    step for displacement and force boundary conditions.

    :arg float max_displacement_rate: The displacement rate in [m] per step
        during the linear phase of the displacement-time graph, and the
        maximum displacement rate of any part of the simulation.
    :arg int steps: The number of simulation steps to conduct.
    :arg float max_displacement: The final applied displacement in [m].
        Default is 0.
    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg float build_load_steps: The inverse of the number of steps
        required to build up to full external force loading.
    :arg float max_load: The maximum total external load in [N] applied to the
        loaded nodes.
    :returns tuple: A tuple of the (steps) displacement_bc_array and the
        (steps) force_bc_array which are the magnitudes applied to the
        displacement and force boundary conditions over each step,
        respectively.
    :rtype tuple (numpy.ndarray, numpy.ndarray):
    """
    # Calculate no. of time steps that applied BCs are in the build phase
    if not ((max_displacement_rate is None)
            or (build_displacement is None)
            or (max_displacement is None)):
        build_time, coefficients = calc_build_time(
            build_displacement, max_displacement_rate, steps)
    else:
        build_time, coefficients = None, None

    displacement_bc_array = []
    force_bc_array = []

    ease_off = 0
    for step in range(steps):
        # Increase external forces in linear incremenets
        force_bc_array.append(
            _increment_load(build_load_steps, max_load, step))

        # Increase displacement in 5th order polynomial increments
        displacement_bc_magnitude, ease_off = _increment_displacement(
            coefficients, build_time, step, ease_off,
            max_displacement_rate, build_displacement, max_displacement)
        displacement_bc_array.append(
            displacement_bc_magnitude)

    return (np.array(displacement_bc_array), np.array(force_bc_array))


def smooth_step_data(start_time_step, final_time_step, start_value,
                     final_value):

    alpha = np.zeros(final_time_step)

    for current_time_step in range(start_time_step, final_time_step):

        xi = (current_time_step - start_time_step) / (final_time_step -
                                                      start_time_step)
        alpha[current_time_step] = (start_value + (final_value - start_value)
                                    * xi**3 * (10 - 15 * xi + 6 * xi**2))

    return alpha