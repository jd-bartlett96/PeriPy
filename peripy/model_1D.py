"""Peridynamics model."""
from numpy.core.numeric import array_equiv
from .integrators import Integrator
from .utilities import write_array
from .create_crack import create_crack
from .correction import (set_volume_correction,
                         set_imprecise_surface_correction,
                         set_precise_surface_correction,
                         set_micromodulus_function)
from collections import namedtuple
import numpy as np
import pathlib
from tqdm import trange
import warnings
import meshio 
import sklearn.neighbors as neighbors


class Model_1D(object):
    def __init__(self, mesh_coords, integrator, horizon, critical_stretch,
                 bond_stiffness, transfinite=0,
                 volume_total=None, write_path=None, connectivity=None,
                 family=None, volume=None, initial_crack=None, dimensions=1,
                 is_density=None, is_bond_type=None,
                 is_displacement_boundary=None, is_force_boundary=None,
                 is_tip=None, density=None, bond_types=None,
                 stiffness_corrections=None,
                 surface_correction=None, volume_correction=None,
                 micromodulus_function=None, node_radius=None):

        """
        Do we need a mesh file for this? Or how should the geometry be defined?
        either use gmsh to generate mesh or could write a bit of code to take a numpy linspace 
        and check type then use this to define the mesh as a point cloud, i.e. no connectivity

        Volume defined in 1D
        just use a unit area and the volume will essentially be the length of the linspace

        Keep bond force as a 2D array, (nnodes, coord)

        Look in geometry files and turn the pythagoras into linear length
        --> in spatial pyx etc

        Work with integrators

        Transfinite unnecessary? --> therefore is volume_total necessary?

        How do family and volume arguments work?

        How does is_bond_type work? And does it need to be ammended to take only 1 positional argument?


        What's going on with is_tip? Is it necessary?
        Unnecessary for now


        How to use cdef and the difference between the C functions and python functions in cython and spatial.pyx?

        Should I collapse the vector bnd down to a single value for 1D? Are the other 2 values just BD in y and z dirns?

        How to deal with self.mesh_boundary in _read_mesh function?

        Do I need to write the coords to memory like in write_mesh function?

        In _set_neighbour_list what is going on with context? Leave as is

        Do I need the _set_surface_corrections function? No

        Does all this plus c stuff just go when using the linearised theory? yes it does go away.

        Ask about the set_tip function.

        Can we make is_tip, is_force_boundary and is_displacement_boundary return
        single value rather than a list of 3 values?

        Dont forget to reinstall peripy every time that the cython code is changed

        What is going on with the mesh.io file? When does it write and read the file?
        What does it do?

        """



        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)
        else:
            self.integrator = integrator

        # If no write path was provided, assign it as None so that model arrays
        # are not written, otherwise, ensure write_path is a Path objects
        if write_path is None:
            self.write_path = None
        else:
            self.write_path = pathlib.Path(write_path)

        # Set model dimensionality 
        self.dimensions = 1

        '''if dimensions == 1:
            self.mesh_elements = _mesh_elements_1d
        else:
            raise DimensionalityError(dimensions)
        #Should be unncessary but could be important later?    
        '''

        # Read coordinates and connectivity from mesh file
        # Want to take this out but need to do some prep first
        self._read_mesh(mesh_coords)
        self.coords_2D = np.array([[coord] for coord in self.coords])

        # Calculate the volume for each node, if None is provided
        # How does the volume concept translate to 1D?
        if volume is None:
            # Calculate the volume for each node
            this_may_take_a_while(self.nnodes, 'volume')
            self.volume = self._set_volumes(volume_total)
            if self.write_path is not None:
                write_array(self.write_path, "volume", self.volume)
        elif type(volume) == np.ndarray:
            if np.shape(volume) != (self.nnodes, ):
                raise ValueError("volume shape is wrong, and must be "
                                 "(nnodes, ) (expected {}, got {})".format(
                                     (self.nnodes, ),
                                     np.shape(volume)))
            warnings.warn(
                    "Reading volume from argument.")
            self.volume = volume.astype(np.float64)
        else:
            raise TypeError("volume type is wrong (expected {}, got "
                            "{})".format(type(volume),
                                         np.ndarray))

        if volume_correction is not None:
            if not ((type(node_radius) == float)
                    or (type(node_radius) == np.float64)):
                raise TypeError(
                    "If volume_correction (= {}) is applied, an "
                    "average node radius must be supplied as the "
                    "keyword argument node_radius. Suggested value"
                    " node_radius = 1. / 2 * np.power(volume_total / "
                    "nnodes, 1. / 3)= {}, "
                    "(expected {}, got {})".format(
                        volume_correction,
                        (1. / 2) * np.power(np.sum(self.volume) / self.nnodes,
                                            1. / 3), float, type(node_radius)))
            # Partial volumes a node radius outside the horizon distance will
            # contribute to the pairwise force function integral, and
            # therefore must be included in the neighbour distance search
            self.horizon = horizon + node_radius
        else:
            self.horizon = horizon

        if volume_correction is not None:
            if not ((type(node_radius) == float)
                    or (type(node_radius) == np.float64)):
                raise TypeError(
                    "If volume_correction (= {}) is applied, an "
                    "average node radius must be supplied as the "
                    "keyword argument node_radius. Suggested value"
                    " node_radius = 1. / 2 * np.power(volume_total / "
                    "nnodes, 1. / 3)= {}, "
                    "(expected {}, got {})".format(
                        volume_correction,
                        (1. / 2) * np.power(np.sum(self.volume) / self.nnodes,
                                            1. / 3), float, type(node_radius)))
            # Partial volumes a node radius outside the horizon distance will
            # contribute to the pairwise force function integral, and
            # therefore must be included in the neighbour distance search
            self.horizon = horizon + node_radius
        else:
            self.horizon = horizon

        # Calculate the family (number of bonds in the initial configuration)
        # and connectivity for each node, if None is provided
        if family is None or connectivity is None:
            # Calculate neighbour list
            this_may_take_a_while(self.nnodes, 'family, connectivity')
            (self.family,
             nlist,
             n_neigh,
             self.max_neighbours) = self._set_neighbour_list(
                 self.coords, self.horizon, 
                 self.nnodes, integrator.context)
            if self.write_path is not None:
                write_array(self.write_path, "family", self.family)
                write_array(self.write_path, "nlist", nlist)
                write_array(self.write_path, "n_neigh", n_neigh)
        else:

            if type(family) == np.ndarray:
                if np.shape(family) != (self.nnodes, ):
                    raise ValueError("family shape is wrong, and must be "
                                     "(nnodes, ) (expected {}, got {})".format(
                                         (self.nnodes, ),
                                         np.shape(family)))
                warnings.warn(
                        "Reading family from argument.")
                self.family = family.astype(np.intc)
            elif type(family) != np.ndarray:
                raise TypeError("family type is wrong (expected {}, got "
                                "{})".format(type(family),
                                             np.ndarray))

            if type(connectivity) == tuple:
                if len(connectivity) != 2:
                    raise ValueError("connectivity size is wrong (expected 2,"
                                     " got {})".format(len(connectivity)))
                warnings.warn(
                    "Reading connectivity from argument.")
                nlist, n_neigh = connectivity
                nlist = nlist.astype(np.intc)
                n_neigh = n_neigh.astype(np.intc)
                if integrator.context is None:
                    self.max_neighbours = np.intc(
                                np.shape(nlist)[1]
                            )
                    if self.max_neighbours != self.family.max():
                        raise ValueError(
                            "max_neighbours, which is equal to the"
                            " size of axis 1 of nlist is wrong (expected "
                            " max_neighbours = np.shape(nlist)[1] = "
                            "family.max() = {}, got {})".format(
                                self.family.max(), self.max_neighbours))
                else:
                    self.max_neighbours = np.intc(
                        np.shape(nlist)[1]
                        )
                    test = self.max_neighbours - 1
                    if self.max_neighbours & test:
                        raise ValueError(
                            "max_neighbours, which is equal to the"
                            " size of axis 1 of nlist is wrong (expected "
                            " max_neighbours = np.shape(nlist)[1] = {},"
                            " got {})".format(
                                1 << (int(self.family.max() - 1)).bit_length(),
                                self.max_neighbours))
            else:
                raise TypeError("connectivity type is wrong (expected {} or"
                                " {}, got {})".format(
                                    tuple, type(None), type(connectivity)))

        if np.any(self.family == 0):
            raise FamilyError(self.family)

        self.initial_connectivity = (nlist, n_neigh)
        self.degrees_freedom = 3

        if connectivity:
            self.mesh_connectivity = connectivity
        else:
            self.mesh_connectivity = False


        # Calculate stiffness corrections if None is provided
        if stiffness_corrections is None:
            stiffness_correction_factors_are_applied = False
            stiffness_corrections = np.ones(
                (self.nnodes, self.max_neighbours), dtype=np.float64)
            if micromodulus_function is not None:
                # Apply micromodulus function
                # Calculate the micromodulus function values
                stiffness_corrections = self._set_micromodulus_values(
                    micromodulus_function, stiffness_corrections, horizon)
                stiffness_correction_factors_are_applied = True
            if volume_correction is not None:
                # Apply volume correction algorithm
                # Calculate the volume correction factors
                stiffness_corrections = self._set_volume_corrections(
                    volume_correction, stiffness_corrections, node_radius,
                    horizon)
                stiffness_correction_factors_are_applied = True
            if surface_correction is not None:
                # Apply surface correction algorithm
                # Calculate the stiffness correction factors
                stiffness_corrections = self._set_surface_corrections(
                    surface_correction, stiffness_corrections)
                stiffness_correction_factors_are_applied = True
            if stiffness_correction_factors_are_applied:
                self.stiffness_corrections = stiffness_corrections
                # Write the stiffness_corrections to file
                if self.write_path is not None:
                    write_array(
                        self.write_path,
                        "stiffness_corrections", stiffness_corrections)
            else:
                # Stiffness corrections factors are not applied
                # This results in speedups in the memory constrained case
                self.stiffness_corrections = None
        elif type(stiffness_corrections) == np.ndarray:
            if np.shape(stiffness_corrections) != (
                    self.nnodes, self.max_neighbours):
                raise ValueError("stiffness_corrections shape is wrong, "
                                 "and must be (nnodes, max_neighbours) "
                                 "(expected {}, got {})".format(
                                     (self.nnodes, self.max_neighbours),
                                     np.shape(stiffness_corrections)))
            else:
                warnings.warn(
                    "Reading stiffness_corrections from argument and "
                    "overriding surface_correction (={}), volume_correction "
                    "(={}) and micromodulus_function (={}) flags! If these "
                    "flags have changed since the stiffness_corrections were "
                    "written to file, please remove the stiffness_corrections "
                    "argument and overwrite existing stiffness_corrections "
                    "with the stiffness_corrections calculated using the new "
                    "flags.".format(
                        surface_correction, volume_correction,
                        micromodulus_function))
                self.stiffness_corrections = (
                    stiffness_corrections.astype(np.float64))
        else:
            raise TypeError("stiffness_corrections type is wrong (expected {}"
                            ", got {})".format(
                                    np.ndarray, type(stiffness_corrections)))

        # Create dummy is_bond_type function if None is provided
        if is_bond_type is None:
            def is_bond_type(x, y):
                return 0

        # Set damage model
        (self.bond_stiffness,
         self.critical_stretch,
         self.plus_cs,
         self.nbond_types,
         self.nregimes) = self._set_damage_model(
             bond_stiffness, critical_stretch)

        if bond_types is None:
            # Calculate bond types and write to file
            self.bond_types = self._set_bond_types(
                self.initial_connectivity, is_bond_type,
                self.nbond_types, self.nregimes)
        elif type(bond_types) == np.ndarray:
            if np.shape(bond_types) != (self.nnodes, self.max_neighbours):
                raise ValueError("bond_types shape is wrong, "
                                 "and must be (nnodes, max_neighbours) "
                                 "(expected {}, got {})".format(
                                     (self.nnodes, self.max_neighbours),
                                     np.shape(bond_types)))
            warnings.warn(
                "Reading bond_types from argument.")
            self.bond_types = bond_types.astype(np.intc)
        else:
            raise TypeError("bond_types type is wrong (expected {}"
                            ", got {})".format(
                                    np.ndarray, type(bond_types)))

        # Set densities of the model
        self.densities = self._set_densities(density, is_density)

        # Create dummy boundary conditions functions if None is provided
        # This could be cut down to a single value in the bnd vectors for 1D
        if is_force_boundary is None:
            def is_force_boundary(x):
                # Node does not live on forces boundary
                bnd = None
                return bnd
        if is_displacement_boundary is None:
            def is_displacement_boundary(x):
                # Node does not live on displacement boundary
                bnd = None
                return bnd
        if is_tip is None:
            def is_tip(x):
                # Node does not live on tip
                bnd = None
                return bnd

        # Apply boundary conditions
        (self.bc_types,
         self.bc_values,
         self.force_bc_types,
         self.force_bc_values,
         self.tip_types,
         self.ntips) = self._set_boundary_conditions(
            is_displacement_boundary, is_force_boundary, is_tip)

        # Build the integrator
        self.integrator.build(
            self.nnodes, self.degrees_freedom, self.max_neighbours,
            self.coords, self.volume, self.family, self.bc_types,
            self.bc_values, self.force_bc_types, self.force_bc_values,
            self.stiffness_corrections, self.bond_types, self.densities)

    #Read the mesh. Not reading from mesh.io but taking an extra attribute. 
    def _read_mesh(self, coords):

        """
        Reads the single dof coords from the attribute coords.

        :arg np.ndarray coords: Single dof coords of the positions of nodes

        :returns: None
        :rtype: NoneType
        """
        
        # Check type of mesh coords (maybe should check shape of array too)
        if type(coords) == np.ndarray:
            self.coords = np.array(coords, dtype=np.float64)
            self.nnodes = self.coords.shape[0]
        else:
            raise TypeError("bond_types type is wrong (expected {}"
                            ", got {})".format(
                                    np.ndarray, type(coords)))

        # Need to deal with boundary here.

    def write_mesh(self, filename, damage=None, displacements=None,
                   file_format=None):
        """
        Write the model's nodes, connectivity and boundary to a mesh file.

        :arg str filename: Path of the file to write the mesh to.
        :arg damage: The damage of each node. Default is None.
        :type damage: :class:`numpy.ndarray`
        :arg displacements: An array with shape (nnodes, dim) where each row is
            the displacement of a node. Default is None.
        :type displacements: :class:`numpy.ndarray`
        :arg str file_format: The file format of the mesh file to
            write. Inferred from `filename` if None. Default is None.

        :returns: None
        :rtype: NoneType
        """
        

        if self.mesh_connectivity is False:
            meshio.write_points_cells(
                filename,
                points=self.coords_2D,
                cells=[],
                point_data={
                    "damage": damage,
                    "displacements": displacements
                    },
                file_format=file_format
                )
        else:
            meshio.write_points_cells(
                filename,
                points=self.coords,
                cells=[
                    (self.mesh_elements.connectivity, self.mesh_connectivity),
                    (self.mesh_elements.boundary, self.mesh_boundary)
                    ],
                point_data={
                    "damage": damage,
                    "displacements": displacements
                    },
                file_format=file_format
                )



    def _set_neighbour_list(self, coords, horizon, nnodes,
                            context=None):
        """
        Build the connectivity and family using a neighbour list.

        Determine the number of nodes within the horizon distance of each node,
        and the neighbour list as a fixed length array. The crack, if it
        exists, is also initiated here. This implementation makes use of
        :meth:`sklearn.neighbors.KDTree`. In preliminary tests, the
        time-expense optimal leaf_size=~160 for 10e5 < nnodes < 10e7.

        :arg coords: The coordinates of all nodes.
        :type coords: :class:`numpy.ndarray`
        :arg float horizon: The horizon distance.
        :arg int nnodes: The number of nodes.
        :arg func initial_crack: The initial crack function, default is None.
        :arg context: The OpenCL context with a single suitable device,
            default is None.
        :type context: :class:`pyopencl._cl.Context` or NoneType

        :returns: An (nnodes,) array of the number of nodes
            within the horizon of each node, the neighbour list as a fixed
            length (nnodes, max_neighbours) array, an (nnodes,) array of the
            number of neighbours of each node at the current time step and
            max_neighbours, the number of columns in nlist.
        :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
                      :class:`numpy.ndarray`, int)
        """
        # Make the coords into a 2D array for the benefit of the KDTree 
        # function. Each second coord is just a 0. TO DO: check this works.
        coords_2D = []
        for coord in coords:
            coords_2D.append([coord,0])
        tree = neighbors.KDTree(coords_2D, leaf_size=160)
        neighbour_list = tree.query_radius(
            coords_2D, r=horizon)
        # Remove identity values, as there is no bond between a node and itself
        neighbour_list = [
            neighbour_list[i][neighbour_list[i] != i]
            for i in range(nnodes)]

        family = [len(neighbour_list[i]) for i in range(nnodes)]
        family = np.array(family, dtype=np.intc)

        #Whats going on here then?
        if context:
            max_neighbours = np.intc(
                1 << (int(family.max() - 1)).bit_length())
            nlist = -1.*np.ones((nnodes, max_neighbours),
                                dtype=np.intc)
        else:
            max_neighbours = family.max()
            nlist = np.zeros((nnodes, max_neighbours), dtype=np.intc)
        for i in range(nnodes):
            nlist[i][:family[i]] = neighbour_list[i]
        nlist = nlist.astype(np.intc)
        n_neigh = family.copy()

        return (family, nlist, n_neigh, max_neighbours)


    def _set_volumes(self, volume_total):  
        """
        Calculate the volume of each node.

        :arg float volume_total: Total volume of the mesh. Must be provided if
            transfinite mode (transfinite=1) is used.

        :returns: Tuple containing an array of volumes for each node.
        :rtype: :class:`numpy.ndarray`
        """
        if volume_total is None:
            raise TypeError("In transfinite mode, a total mesh volume "
                            "volume_total' must be provided as a keyword"
                            " argument (expected {}, got {})".format(
                                    float, type(volume_total)))        
        tmp = volume_total / self.nnodes
        volume = tmp * np.ones(self.nnodes)
        volume = volume.astype(np.float64)
        return volume

    
    def _set_densities(self, density, is_density):
        """
        Build densities array.

        :arg density: An (nnodes, ) array of node density values, each
            corresponding to a material, or a float value of the density
            if nmaterials=1.
        :type density: :class:`numpy.ndarray`
        :arg is_density: A function that returns a float of the material
            density, given a node coordinate as input.
        :type is_density: function

        :returns: A (nnodes, degrees_freedom) array of nodal densities, or
            None if no is_density function or density array is supplied.
        :rtype: :class:`numpy.ndarray` or None
        """
        if density is None:
            if is_density is None:
                densities = None
            else:
                if not callable(is_density):
                    raise TypeError(
                        "is_density must be a *function*.")
                elif type(is_density(self.coords[0])) is not float:
                    raise TypeError(
                        "is_density must be a function that returns a *float* "
                        "(expected {}, got {})".format(
                            float, type(
                                is_density(self.coords[0]))))
                density = np.ones(self.nnodes)
                for i in range(self.nnodes):
                    density[i] = is_density(self.coords[i])
                if self.write_path is not None:
                    write_array(self.write_path, "density", density)
                densities = np.transpose(
                    np.tile(density, (self.degrees_freedom, 1))).astype(
                        np.float64)
        elif type(density) == np.ndarray:
            if np.shape(density) != (self.nnodes,):
                raise ValueError("densty shape is wrong, and must be "
                                 "(nnodes,) (expected {}, got {})".format(
                                     (self.nnodes,), np.shape(density)))
            warnings.warn(
                "Reading density from argument.")
            densities = np.transpose(
                np.tile(density, (self.degrees_freedom, 1))).astype(np.float64)
        else:
            raise TypeError("density type is wrong, and must be an array of"
                            " shape (nnodes,) (expected {}, got {})".format(
                                np.ndarray, type(density)))
        return densities

    
    def _set_bond_types(self, connectivity, is_bond_type, nbond_types,
                        nregimes):
        """
        Build bond_types array.

        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            None the connectivity at the time of construction of the
            :class:`Model` object will be used. Default None.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg is_bond_type: A function that returns an integer value (a
            flag) of the bond type, given two node coordinates as input.
        :type is_bond_type: function
        :arg int nbond_types: The number of different bonds; the number of
        damage models, for example there might be a damage model for each
        material and interface in a composite.
        :arg int nbond_types: The expected number of regimes in the damage
            model.

        :returns: A (`nnodes`, `max_neighbours`) array of the bond types,
            which are used to index into the bond_stiffness and
            critical_stretch arrays.
        :rtype: :class:`numpy.ndarray`
        """
        if nbond_types != 1:
            if not callable(is_bond_type):
                raise TypeError(
                    "is_bond_type must be a *function*.")
            nlist, n_neigh = connectivity
            bond_types = np.zeros(
                (self.nnodes, self.max_neighbours))
            for i in range(self.nnodes):
                for neigh in range(n_neigh[i]):
                    j = nlist[i][neigh]
                    bond_type = is_bond_type(
                        self.coords[i, :], self.coords[j, :])
                    if type(bond_type) is not int:
                        raise TypeError(
                            "is_bond_type must be a function that returns an "
                            "*int* (expected {}, got {})".format(
                                int, type(bond_type)))
                    if bond_type < 0:
                        raise ValueError(
                            "is_bond_type must be a function that returns a "
                            "*positive* int or 0 (got {})".format(
                                bond_type))
                    if bond_type > nbond_types - 1:
                        raise ValueError(
                            "is_bond_type must be a function that returns a "
                            "positive int or 0 which is *less* than "
                            "nbond_types (the number of different bonds, "
                            "nbond_types = {}, got is_bond_type = {} for "
                            "node coordinate pair {}, {})".format(
                                nbond_types,
                                bond_type,
                                self.coords[i, :],
                                self.coords[j, :]
                                ))
                    bond_types[i][neigh] = bond_type
            bond_types = bond_types.astype(np.intc)
            if self.write_path is not None:
                write_array(self.write_path, "bond_types", bond_types)
        elif nregimes != 1:
            bond_types = np.zeros(
                (self.nnodes, self.max_neighbours))
            bond_types = bond_types.astype(np.intc)
            if self.write_path is not None:
                write_array(self.write_path, "bond_types", bond_types)
        else:
            bond_types = None
        return bond_types


    def _set_micromodulus_values(
            self, micromodulus_function, stiffness_corrections, horizon):
        """
        Calculate an array of normalised micromodulus function values.

        Calculates the micromodulus function values as an array of stiffness
        correction factors. If micromodulus_function = None, a constant
        micromodulus function is applied. If micromodulus_function = 0, a
        conical micromodulus function is applied.

        The correction factors are element-wise multiplied to the existing
        corrections that apply to the bond stiffnesses so that the correction
        factors can be stored in memory as a single array.

        :arg int micromodulus_function: A flag variable. For the micromodulus
            function algorithm.
            Set to None: . Set to 1: . Set to 0:. Default None.
        :arg stiffness_corrections: An (`nnodes`, `max_neighbours`) container
            for the stiffness correction factors for each bond.
        :type stiffness_corrections: :class:`numpy.ndarray`
        :arg float horizon: The peridynamic horizon distance.

        :returns: An (`nnodes`, `max_neighbours`) array of the normalised
            micromodulus function values as an array of stiffness
            correction factor of each bond.
        :rtype: :class:`numpy.ndarray`
        """
        nlist, n_neigh = self.initial_connectivity

        if micromodulus_function == 0:
            # Conical micromodulus function
            set_micromodulus_function(
                stiffness_corrections, self.coords, nlist, n_neigh,
                np.float64(horizon), np.intc(micromodulus_function))
        else:
            raise ValueError("micromodulus_function value is wrong "
                             "(expected 0 or None, got {})".format(
                                 micromodulus_function))

        return stiffness_corrections


    def _set_damage_model(self, bond_stiffness, critical_stretch):
        """
        Calculate the parameters for the damage models.

        Calculates the `+ c`s (c.f. `y = mx + c`) for the n-linear
        damage model for each bond type, where n is nregimes is the number of
        linear splines that define the damage model (e.g. An n-linear damage
        model has nregimes = n. The bond-based prototype microelastic brittle
        (PMB) model has nregimes = 1.)

        :arg bond_stiffness: An (nregimes, nbond_types) array of bond
            stiffness values, each corresponding to a bond type and a regime.
        :type bond_stiffness: list or :class:`numpy.ndarray`
        :arg critical_stretch: An (n_regimes, nbond_types) array of critical
            stretch values, each corresponding to a bond type and a regime.
        :type critical_stretch: list or :class:`numpy.ndarray`

        :raises DamageModelError: when an unsorted (i.e. not in ascending
            order) `critical_stretch` argument is provided.

        :returns: A tuple of the damage model:
            bond_stiffness, a float or array of the bond stiffness(es);
            critcal_stretch, a float or array of the critical stretch(es);
            nbond_types, the number of bond types (damage models) in the model;
            nregimes, the number of `regimes` in the damage model. e.g.
                linear has n_regimes = 1, bi-linear has n_regimes = 2, etc;
            plus_cs, an (`nregimes`, `nbond_types`) array of the `+cs` for each
                linear part of the bond damage models for each bond type. Takes
                a value of None if the PMB (Prototype Micro-elastic Brittle)
                model is used, i.e. n_regimes = 1.
        :rtype: tuple(:class:`numpy.ndarray` or :class:`numpy.float64`,
                      :class:`numpy.ndarray` or :class:`numpy.float64`,
                      :class:`numpy.intc`,
                      :class:`numpy.intc`,
                      :class:`numpy.ndarray` or NoneType)
        """
        if type(bond_stiffness) != type(critical_stretch):
            raise TypeError(
                "bond_stiffness must be the same type "
                "as critical_stretch (expected {}, got {})".format(
                    type(critical_stretch),
                    type(bond_stiffness)))
        if ((type(bond_stiffness) is list) or
                (type(bond_stiffness) is np.ndarray)):
            if np.shape(bond_stiffness) != np.shape(critical_stretch):
                raise ValueError(
                    "The shape of bond_stiffness must be equal to the shape "
                    "of critical_stretch (expected {}, got {})".format(
                        np.shape(critical_stretch),
                        np.shape(bond_stiffness)))
            else:
                if np.shape(bond_stiffness) == (1,):
                    nregimes = 1
                    nbond_types = 1
                    bond_stiffness = np.float64(bond_stiffness[0])
                    critical_stretch = np.float64(critical_stretch[0])
                    plus_cs = None
                elif np.shape(bond_stiffness) == ():
                    nregimes = 1
                    nbond_types = 1
                    bond_stiffness = np.float64(bond_stiffness)
                    critical_stretch = np.array(critical_stretch)
                    plus_cs = None
                elif np.shape(bond_stiffness[0]) == (1,):
                    nregimes = 1
                    nbond_types = np.shape(bond_stiffness)[0]
                    bond_stiffness = np.array(
                        bond_stiffness, dtype=np.float64)
                    critical_stretch = np.array(
                        critical_stretch, dtype=np.float64)
                    plus_cs = np.zeros(nbond_types)
                    plus_cs = plus_cs.astype(np.float64)
                elif np.shape(bond_stiffness[0]) == ():
                    nbond_types = 1
                    nregimes = np.shape(bond_stiffness)[0]

                    if not all(
                        critical_stretch[i] <= critical_stretch[i + 1]
                            for i in range(nregimes-1)):
                        raise DamageModelError(critical_stretch)

                    bond_stiffness = np.array(
                        bond_stiffness, dtype=np.float64)
                    critical_stretch = np.array(
                        critical_stretch, dtype=np.float64)
                    plus_cs = np.zeros(nregimes)
                    c_i = 0.0
                    # The bond force density at 0 stretch is 0
                    plus_cs[0] = c_i
                    c_prev = c_i
                    for r in range(1, nregimes):
                        c_i = (
                            c_prev
                            + (bond_stiffness[r - 1] *
                                critical_stretch[r - 1])
                            - (bond_stiffness[r] *
                               critical_stretch[r - 1]))
                        plus_cs[r] = c_i
                        c_prev = c_i
                    plus_cs = plus_cs.astype(np.float64)
                else:
                    nregimes = np.shape(bond_stiffness)[1]
                    nbond_types = np.shape(bond_stiffness)[0]

                    for i in range(nbond_types):
                        if not all(
                            critical_stretch[i][j] <=
                                critical_stretch[i][j + 1]
                                for j in range(nregimes-1)):
                            raise DamageModelError(critical_stretch[i])

                    bond_stiffness = np.array(
                        bond_stiffness, dtype=np.float64)
                    critical_stretch = np.array(
                        critical_stretch, dtype=np.float64)
                    plus_cs = np.zeros((nbond_types, nregimes))
                    c_i = np.zeros(nbond_types)
                    # The bond force density at 0 stretch is 0
                    plus_cs[:, 0] = c_i
                    c_prev = c_i
                    for r in range(1, nregimes):
                        c_i = (
                            c_prev
                            + (bond_stiffness[:, r - 1] *
                                critical_stretch[:, r - 1])
                            - (bond_stiffness[:, r] *
                               critical_stretch[:, r - 1]))
                        plus_cs[:, r] = c_i
                        c_prev = c_i
                    plus_cs = plus_cs.astype(np.float64)
        elif ((type(bond_stiffness) is float) or
              (type(bond_stiffness) is np.float64)):
            nregimes = 1
            nbond_types = 1
            bond_stiffness = np.float64(bond_stiffness)
            critical_stretch = np.float64(critical_stretch)
            plus_cs = None
        else:
            raise TypeError(
                "Type of bond_stiffness and critical_stretch is not supported "
                "(expected {} or {}, got {})".format(
                    float, np.ndarray, type(bond_stiffness)))

        # Convert to types that OpenCL can handle
        nregimes = np.intc(nregimes)
        nbond_types = np.intc(nbond_types)
        if np.any(critical_stretch < 0):
            raise ValueError("critical_stretch values must not be < 0, "
                             "(got {})".format(critical_stretch))
        return (bond_stiffness, critical_stretch, plus_cs, nbond_types,
                nregimes)

    
    def _set_boundary_conditions(
            self, is_displacement_boundary, is_force_boundary, is_tip):
        """
        Build the boundary conditions of the model.

        :arg is_displacement_boundary: A function to determine if a node is on
            the boundary for a displacement boundary condition, and if it is,
            which direction and magnitude the boundary conditions are applied
            (positive or negative cartesian direction). It has the form
            is_displacement_boundary(:class:`numpy.ndarray`). The argument is
            the initial coordinates of a node being simulated.
            `is_displacement_boundary` returns a (3) list of the boundary types
            in each cartesian direction.
            A boundary type with an int value of None if the node is not
            on a displacement controlled boundary, a value of 1 if is is on a
            boundary and loaded in the positive cartesian direction, and a
            value of -1 if it is on the boundary and loaded in the negative
            direction, and a value of 0 if it is not loaded.
        :type is_displacement_boundary: function
        :arg is_force_boundary: As 'is_displacement_boundary' but applying to
            force boundary conditions as opposed to displacement boundary
            conditions.
        :type is_force_boundary: function
        :arg is_tip: A function to determine if a node is to be measured for
            its state variables or reaction force over time, and if it is,
            which cartesian direction the measurements are made. It has the
            form is_tip(:class:`numpy.ndarray`). The argument is the initial
            coordinates of a node being simulated. `is_tip` returns a
            (3) list of the tip types in each cartesian direction:
            A value of None if the node is not on the `tip`, and a value
            of not None (e.g. a string or an int) if it is on the `tip`
            and to be measured.
        :type is_tip: function

        :returns: A tuple of the displacement and foce boundary condition types
            and values, and the list (nnodes, 3) of tip types, and a dictionary
            of the number of nodes residing on each tip.
        :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
                      :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                      list, dict)
        """
        functions = {'is_displacement_boundary': is_displacement_boundary,
                     'is_force_boundary': is_force_boundary,
                     'is_tip': is_tip}
        for function in functions:
            if not callable(functions[function]):
                raise TypeError("{} must be a *function*.".format(function))
            if (functions[function](self.coords[0]) and
                type(functions[function](self.coords[0])) is not int):
                raise TypeError(
                    "{} must be a function that returns a *list*. Expected "
                    "{} or {}, received {}".format(function, int, None,
                      type(functions[function](self.coords[0]))))
    

        def set_tip(tip, i, tip_types, ntips):
            #Have reduced one parameter from here since we know what
            #direction the tip is in. Might not even be a necessary function
            #
            """Set tip_types dict and ntips dict for this tip."""
            if str(tip) not in ntips:
                # Initiate container for number of nodes
                # residing on this tip and list for tuples of nodes
                ntips[str(tip)] = 0
                tip_types[str(tip)] = []
            # Increase the number of nodes residing
            # on this tip by one
            ntips[str(tip)] += 1
            # Add node and direction to tip types dict
            tip_types[str(tip)].append((i))
            return tip_types, ntips

        bc_types = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.intc)
        bc_values = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.float64)
        force_bc_types = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.intc)
        force_bc_values = np.zeros(
            (self.nnodes, self.degrees_freedom), dtype=np.float64)
        tip_types = {}
        num_force_bc_nodes = 0
        ntips = {'model': self.nnodes}

        #Here it use to loop through the 3 directions, we only have the single
        #but have reduced it down to single direction. Possible error.
        for i in range(self.nnodes):
            bnd = is_displacement_boundary(self.coords[i])
            forces_bnd = is_force_boundary(self.coords[i])
            tip = is_tip(self.coords[i])
            is_force_node = 0
           
            forces_bnd = forces_bnd
            bnd = bnd
            tip = tip
            # Define boundary types and values
            if bnd is not None:
                bc_types[i] = np.intc(1)
                bc_values[i] = np.float64(bnd)
            # Define forces boundary types and values
            if forces_bnd is not None:
                is_force_node = 1
                force_bc_types[i] = np.intc(1)
                force_bc_values[i] = np.float64(
                    forces_bnd / self.volume[i])

            if tip is not None:
                if type(tip) is tuple:
                    for tip_jk in tip:
                        tip_types, ntips = set_tip(
                            tip_jk, i, tip_types, ntips)
                else:
                    tip_types, ntips = set_tip(
                        tip, i, tip_types, ntips)

            num_force_bc_nodes += is_force_node
        if num_force_bc_nodes != 0:
            force_bc_values = np.float64(
                np.divide(force_bc_values, num_force_bc_nodes))

        return (bc_types, bc_values, force_bc_types, force_bc_values,
                tip_types, ntips)

    

    def simulate(self, steps, u=None, ud=None, connectivity=None,
                 regimes=None, critical_stretch=None, bond_stiffness=None,
                 displacement_bc_magnitudes=None, force_bc_magnitudes=None,
                 first_step=1, write=None,
                 write_path=None):
        """
        Simulate the peridynamics model.

        :arg int steps: The number of simulation steps to conduct.
        :arg u: The initial displacements for the simulation. If None the
            displacements will be initialised to zero. Default None.
        :type u: :class:`numpy.ndarray`
        :arg ud: The initial velocities for the simulation. If None the
            velocities will be initialised to zero. Default None.
        :type ud: :class:`numpy.ndarray`
        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            None the connectivity at the time of construction of the
            :class:`Model` object will be used. Default None.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg regimes: The initial regimes for the simulation. A
            (`nodes`, `max_neighbours`) array of type
            :class:`numpy.ndarray` of the regimes of the bonds.
        :type regimes: :class:`numpy.ndarray`
        :arg critical_stretch: An (nregimes, nbond_types) array of critical
            stretch values, each corresponding to a bond type and a regime,
            or a float value of the critical stretch of the Peridynamic
            bond-based prototype microelastic brittle (PMB) model.
        :type critical_stretch: :class:`numpy.ndarray` or float
        :arg bond_stiffness: An (nregimes, nbond_types) array of bond
            stiffness values, each corresponding to a bond type and a regime,
            or a float value of the bond stiffness the Peridynamic bond-based
            prototype microelastic brittle (PMB) model.
        :type bond_stiffness: :class:`numpy.ndarray` or float
        :arg displacement_bc_magnitudes: (steps, ) array of the magnitude
            applied to the displacement boundary conditions over time.
        :type displacement_bc_magnitudes: :class:`numpy.ndarray`
        :arg force_bc_magnitudes: (steps, ) array of the magnitude applied to
            the force boundary conditions over time.
        :type force_bc_magnitudes: :class:`numpy.ndarray`
        :arg int first_step: The starting step number. This is useful when
            restarting a simulation.
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling :meth:`Model.write_mesh`. If None then
            no output is written. Default None.
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str

        :returns: A tuple of the final displacements (`u`); damage,
            a tuple of the connectivity; the final node forces (`force`);
            the final node velocities (`ud`) and a dictionary object
            containing the displacement, velocity and acceleration
            (average of), and the forces and body forces
            for each of the writes (read 'over time'), for each unique
            tip_type (read 'for each of the set of nodes the user has
            chosen to measure datum for, as defined by the `is_tip` function).
        :rtype: tuple(
            :class:`numpy.ndarray`, :class:`numpy.ndarray`,
            tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`),
            :class:`numpy.ndarray`, :class:`numpy.ndarray`,
            dict)
        """
        (u,
         ud,
         udd,
         force,
         body_force,
         nlist,
         n_neigh,
         displacement_bc_magnitudes,
         force_bc_magnitudes,
         damage,
         data,
         nwrites,
         write_path) = self._simulate_initialise(
             steps, first_step, write, regimes, u, ud,
             displacement_bc_magnitudes, force_bc_magnitudes, connectivity,
             bond_stiffness, critical_stretch, write_path)

        for step in trange(first_step, first_step+steps,
                           desc="Simulation Progress", unit="steps"):

            # Call one integration step
            self.integrator(
                displacement_bc_magnitudes[step - 1],
                force_bc_magnitudes[step - 1])

            if write:
                if step % write == 0:
                    (u,
                     ud,
                     udd,
                     force,
                     body_force,
                     damage,
                     nlist,
                     n_neigh) = self.integrator.write(
                         u, ud, udd, body_force, force, damage, nlist, n_neigh)

                    self.write_mesh(write_path/f"U_{step}.vtk", damage, u)

                    # Write index number
                    ii = step // write - (first_step - 1) // write - 1

                    for tip_type, node_list in self.tip_types.items():
                        if tip_type not in data:
                            # Build data dict for this tip type
                            data[tip_type] = {
                                'displacement': np.zeros(
                                    nwrites, dtype=np.float64),
                                'velocity': np.zeros(
                                    nwrites, dtype=np.float64),
                                'acceleration': np.zeros(
                                    nwrites, dtype=np.float64),
                                'force': np.zeros(
                                    nwrites, dtype=np.float64),
                                'body_force': np.zeros(
                                    nwrites, dtype=np.float64)
                                }
                        for node in node_list:
                            i, j = node
                            # Add to tip data for the write index, ii
                            data[tip_type]['displacement'][ii] += (
                                u[i, j])
                            data[tip_type]['velocity'][ii] += (
                                ud[i, j])
                            data[tip_type]['acceleration'][ii] += (
                                udd[i, j])
                            data[tip_type]['force'][ii] += (
                                force[i, j] * self.volume[i])
                            data[tip_type]['body_force'][ii] += (
                                body_force[i, j] * self.volume[i])

                    # Add to model data for the write index, ii
                    data['model']['step'][ii] = step
                    data['model']['displacement'][ii] = np.sum(u)
                    data['model']['velocity'][ii] = np.sum(ud)
                    data['model']['acceleration'][ii] = np.sum(udd)
                    data['model']['force'][ii] = np.sum(
                        force * self.volume[:, np.newaxis])
                    data['model']['body_force'][ii] = np.sum(
                        body_force * self.volume[:, np.newaxis])

                    damage_sum = np.sum(damage)
                    data['model']['damage_sum'][ii] = damage_sum
                    if damage_sum > 0.05*self.nnodes:
                        warnings.warn('Over 5% of bonds have broken!\
                                      peridynamics simulation continuing')
                    elif damage_sum > 0.7*self.nnodes:
                        warnings.warn('Over 7% of bonds have broken!\
                                      peridynamics simulation continuing')
        for tip_type_str in data:
            # Average the nodal displacements, velocities and
            # accelerations
            ntip = self.ntips[tip_type_str]
            if ntip != 0:
                data[tip_type_str]['displacement'] /= ntip
                data[tip_type_str]['velocity'] /= ntip
                data[tip_type_str]['acceleration'] /= ntip
        (u,
         ud,
         udd,
         force,
         body_force,
         damage,
         nlist,
         n_neigh) = self.integrator.write(
             u, ud, udd, force, body_force, damage, nlist, n_neigh)

        return (u, damage, (nlist, n_neigh), force, ud, data)

 
    def _simulate_initialise(
            self, steps, first_step, write, regimes, u, ud,
            displacement_bc_magnitudes, force_bc_magnitudes, connectivity,
            bond_stiffness, critical_stretch, write_path):
        """
        Initialise simulation variables.

        :arg int steps: The number of simulation steps to conduct.
        :arg int first_step: The starting step number. This is useful when
            restarting a simulation.
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling :meth:`Model.write_mesh`. If None then
            no output is written. Default None.
        :arg regimes: The initial regimes for the simulation. A
            (`nodes`, `max_neighbours`) array of type
            :class:`numpy.ndarray` of the regimes of the bonds.
        :type regimes: :class:`numpy.ndarray`
        :arg u: The initial displacements for the simulation. If None the
            displacements will be initialised to zero. Default None.
        :type u: :class:`numpy.ndarray`
        :arg ud: The initial velocities for the simulation. If None the
            velocities will be initialised to zero. Default None.
        :type ud: :class:`numpy.ndarray`
        :arg displacement_bc_magnitudes: (steps, ) array of the magnitude
            applied to the displacement boundary conditions over time.
        :type displacement_bc_magnitudes: :class:`numpy.ndarray`
        :arg force_bc_magnitudes: (steps, ) array of the magnitude applied to
            the force boundary conditions over time.
        :type force_bc_magnitudes: :class:`numpy.ndarray`
        :arg connectivity: The initial connectivity for the simulation. A tuple
            of a neighbour list and the number of neighbours for each node. If
            None the connectivity at the time of construction of the
            :class:`Model` object will be used. Default None.
        :type connectivity: tuple(:class:`numpy.ndarray`,
            :class:`numpy.ndarray`)
        :arg bond_stiffness: An (nregimes, nbond_types) array of bond
            stiffness values, each corresponding to a bond type and a regime.
        :type bond_stiffness: list or :class: `numpy.ndarray`
        :arg critical_stretch: An (nregimes, nbond_types) array of critical
            stretch values, each corresponding to a bond type and a regime.
        :type critical_stretch: list or :class: `numpy.ndarray`
        :arg write_path: The path where the periodic mesh files should be
            written.
        :type write_path: path-like or str

        :returns: A tuple of initialised variables used for simulation.
        :type: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, :class:`numpy.ndarray`,
                     :class:`numpy.ndarray`, dict,
                     int, :class`pathlib.Path`)
        """
        # Create initial displacements and velocities if None is provided
        #TO DO: check this works, changed u from 1D to 3D
        if u is None:
            u = np.zeros((self.nnodes), dtype=np.float64)
        if ud is None:
            ud = np.zeros((self.nnodes), dtype=np.float64)
        # Initiate forces, damage and accelerations
        force = np.zeros((self.nnodes), dtype=np.float64)
        body_force = np.zeros((self.nnodes), dtype=np.float64)
        damage = np.zeros(self.nnodes, dtype=np.float64)
        udd = np.zeros((self.nnodes), dtype=np.float64)
        # Create boundary condition magnitudes if None is provided
        if displacement_bc_magnitudes is None:
            displacement_bc_magnitudes = np.zeros(
                first_step + steps - 1, dtype=np.float64)
        elif type(displacement_bc_magnitudes) == np.ndarray:
            if len(displacement_bc_magnitudes) < steps:
                raise ValueError("displacement_bc_magnitudes length must be "
                                 "equal to or greater than (first_step + steps"
                                 " - 1), (expected {}, got {})".format(
                                     first_step + steps - 1,
                                     len(displacement_bc_magnitudes)))
            displacement_bc_magnitudes = displacement_bc_magnitudes.astype(
                np.float64)
        else:
            raise TypeError("displacement_bc_magnitudes type is wrong "
                            "(expected {}, got {})".format(
                                np.ndarray,
                                type(displacement_bc_magnitudes)))
        if force_bc_magnitudes is None:
            force_bc_magnitudes = np.zeros(
                first_step + steps - 1, dtype=np.float64)
        elif type(force_bc_magnitudes) == np.ndarray:
            if len(force_bc_magnitudes) < steps:
                raise ValueError("force_bc_magnitudes length must be "
                                 "equal to or greater than (first_step + steps"
                                 " - 1), (expected {}, got {})".format(
                                     first_step + steps - 1,
                                     len(force_bc_magnitudes)))
            force_bc_magnitudes = force_bc_magnitudes.astype(
                np.float64)
        else:
            raise TypeError("force_bc_magnitudes type is wrong "
                            "(expected {}, got {})".format(
                                np.ndarray,
                                type(force_bc_magnitudes)))
        # Use the initial connectivity (when the Model was constructed) if none
        # is provided
        if connectivity is None:
            nlist, n_neigh = self.initial_connectivity
            # Make a copy so that the initial_connectivity remains unchanged
            nlist = nlist.copy()
            n_neigh = n_neigh.copy()
        elif type(connectivity) == tuple:
            if len(connectivity) != 2:
                raise ValueError("connectivity size is wrong (expected 2,"
                                 " got {})".format(len(connectivity)))
            nlist, n_neigh = connectivity
        else:
            raise TypeError("connectivity type is wrong (expected {} or"
                            " {}, got {})".format(
                                    tuple, type(None), type(connectivity)))
        # Use the initial regimes of linear elastic (0 values) if none
        # is provided
        if regimes is None:
            regimes = np.zeros(
                (self.nnodes, self.max_neighbours), dtype=np.intc)
        elif type(regimes) == np.ndarray:
            if np.shape(regimes) != (self.nnodes, self.max_neighbours):
                raise ValueError("regimes shape is wrong, and must be "
                                 "(nnodes, max_neighbours) "
                                 "(expected {}, got {})".format(
                                     (self.nnodes, self.max_neighbours),
                                     np.shape(regimes)))
            regimes = regimes.astype(np.intc)
        else:
            raise TypeError("regimes type is wrong "
                            "(expected {} or {}, got {})".format(
                                np.ndarray,
                                type(None),
                                type(regimes)))

        # Use the initial damage model
        # (when the Model was constructed) if None is provided
        if (bond_stiffness is None) and (critical_stretch is None):
            bond_stiffness = self.bond_stiffness
            critical_stretch = self.critical_stretch
            plus_cs = self.plus_cs
            nbond_types = self.nbond_types
            nregimes = self.nregimes
        else:
            (bond_stiffness,
             critical_stretch,
             plus_cs,
             nbond_types,
             nregimes) = self._set_damage_model(
                 bond_stiffness, critical_stretch)
            if nbond_types != self.nbond_types:
                raise ValueError(
                    "Number of bond types has unexpectedly changed from when "
                    " the model was constructed. Please reinstantiate "
                    ":class:`Model` with the new number of bond types "
                    "(expected {}, got {}).".format(
                        self.nbond_types, nbond_types))

        # If no write path was provided use the current directory, otherwise
        # ensure write_path is a Path object.
        if write_path is None:
            write_path = pathlib.Path()
        else:
            write_path = pathlib.Path(write_path)

        # Container for plotting data
        data = {}
        nwrites = None
        if write:
            nwrites = (
                (first_step + steps - 1) // write - (first_step - 1) // write)
            if write is not None:
                data['model'] = {
                    'step': np.zeros(nwrites, dtype=int),
                    'displacement': np.zeros(nwrites, dtype=np.float64),
                    'velocity': np.zeros(nwrites, dtype=np.float64),
                    'acceleration': np.zeros(nwrites, dtype=np.float64),
                    'force': np.zeros(nwrites, dtype=np.float64),
                    'body_force': np.zeros(nwrites, dtype=np.float64),
                    'damage_sum': np.zeros(nwrites, dtype=np.float64)
                    }

        # Initialise the OpenCL buffers
        self.integrator.create_buffers(
            nlist, n_neigh, bond_stiffness, critical_stretch, plus_cs, u, ud,
            udd, force, body_force, damage, regimes, nregimes, nbond_types)

        return (u, ud, udd, force, body_force, nlist, n_neigh,
                displacement_bc_magnitudes, force_bc_magnitudes, damage, data,
                nwrites, write_path)













def this_may_take_a_while(nnodes, calculation):
    """
    Raise a UserWarning if nnodes is over 9000! (an arbritrarily large value).

    :arg int nnodes: The number of nodes.
    :arg str calculation: A string message of the calculation being performed.

    :rtype: :class:`UserWarning` or None
    """
    message = (
        f"Calculating {calculation} for {nnodes} nodes... "
        "this may take a while."
        )

    if nnodes > 9000:
        warnings.warn(message)


class DimensionalityError(Exception):
    """An invalid dimensionality argument used to construct a model."""

    def __init__(self, dimensions):
        """
        Construct the exception.

        :arg int dimensions: The number of dimensions passed as an argument to
            :meth:`Model`.

        :rtype: :class:`DimensionalityError`
        """
        message = (
                "The number of dimensions must be 2 or 3,"
                f" {dimensions} was given."
                )

        super().__init__(message)


class FamilyError(Exception):
    """One or more nodes have no bonds in the initial state."""

    def __init__(self, family):
        """
        Construct the exception.

        :arg family: The family array.
        :type family: :class:`numpy.ndarray`

        :rtype: :class:`FamilyError`
        """
        indicies = np.where(family == 0)[0]
        indicies = " ".join([f"{index}" for index in indicies])
        message = (
                "The following nodes have no bonds in the initial state,"
                f" {indicies}."
                )

        super().__init__(message)


class DamageModelError(Exception):
    """An invalid critical stretch argument was used to construct a model."""

    def __init__(self, critical_stretch):
        """
        Construct the exception.

        :arg critical_stretch: The critical_stretch array.
        :type critical_stretch: :class:`numpy.ndarray` or list

        :rtype: :class:`DamageModelError`
        """
        message = (
                "The critical_stretch list or array for a bond-type with "
                "multiple regimes must be in ascending order, "
                f" {critical_stretch} was given."
                )

        super().__init__(message)


class InvalidIntegrator(Exception):
    """An invalid integrator has been passed to `simulate`."""

    def __init__(self, integrator):
        """
        Construct the exception.

        :arg integrator: The object passed to :meth:`Model.simulate` as the
            integrator argument.

        :rtype: :class:`InvalidIntegrator`
        """
        message = (
                f"{integrator} is not an instance of"
                "peripy.integrators.Integrator"
                )

        super().__init__(message)
