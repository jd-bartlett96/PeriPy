from libc.math cimport sqrt


def euclid(r1, r2):
    """
    Calculate the Euclidean distance between two, three-dimensional
    coordinates.
    :arg r1: A (3,) array representing the first coordinate.
    :type r1: :class:`numpy.ndarray`
    :arg r2: A (3,) array representing the second coordinate.
    :type r2: :class:`numpy.ndarray`
    :returns: The Euclidean distance between `r1` and `r2`
    :rtype: `numpy.float64`
    """
    return ceuclid(r1, r2)


cdef inline double ceuclid(double[:] r1, double[:] r2):
    """
    C function for calculating the Euclidean distance between two coordinates.
    """
    cdef double[1] dr

    dr[0] = r2[0] - r1[0]
    
    return dr[0]


def strain(r1, r2, r10, r20):
    """
    Calculate the strain between two particles given their current and initial
    positions.
    :arg r1: A (3,) array giving the coordinate of the first particle.
    :type r1: :class:`numpy.ndarray`
    :arg r2: A (3,) array giving the coordinate of the second particle.
    :type r2: :class:`numpy.ndarray`
    :arg r10: A (3,) array giving the initial coordinate of the first particle.
    :type r10: :class:`numpy.ndarray`
    :arg r20: A (3,) array giving the initial coordinate of the second particle.
    :type r20: :class:`numpy.ndarray`
    :returns: The strain.
    :rtype: `numpy.float64`
    """
    return cstrain(r1, r2, r10, r20)


cdef inline double cstrain(double[:] r1, double[:] r2,
                           double[:] r10, double[:] r20):
    """
    C function for calculating the strain given current and initial
    coordinates.
    """
    cdef double l, dl

    l = ceuclid(r1, r2)
    l0 = ceuclid(r10, r20)
    dl = l - l0

    return dl/l0


def strain2(l, r10, r20):
    """
    Calculate the strain between two particles given their current distance and
    initial positions.
    :arg l: The Euclidean distance between the two particles.
    :type l: `numpy.float64`
    :arg r10: A (3,) array giving the initial coordinate of the first particle.
    :type r10: :class:`numpy.ndarray`
    :arg r20: A (3,) array giving the initial coordinate of the second particle.
    :type r20: :class:`numpy.ndarray`
    :returns: The strain.
    :rtype: `numpy.float64`
    """
    return cstrain2(l, r10, r20)


cdef inline double cstrain2(double l, double[:] r10, double[:] r20):
    """
    C function for calculating the strain given the current displacement and
    the initial coordinates.
    """
    cdef double dl

    l0 = ceuclid(r10, r20)
    dl = l - l0

    return dl/l0