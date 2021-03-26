# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')

def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    #check if the dimension of the cell is 1
    if cell.dim == 1:
        #for an interval we can simply calculate the distance between each point by
        #taking the length of the interval and dividing by the degree of our polynomial
        between_points = (cell.vertices[1] - cell.vertices[0])/degree
        #initialize a list with the lower bound of our interval as the first element
        points = [np.array(cell.vertices[0])]
        #a for loop that appends a new element to the list, which consists of the previous element
        #added + the distance between points, and stores it into the list as a np.array
        for i in range(1, degree+1):
            points.append(np.array(points[-1]+between_points))
        #return an np array with our 1-vectors we just calculated
        return np.array(points)
    #check if the dimension of the cell is 2
    if cell.dim == 2:
        #initialize an empty list
        points = []
        #loop over a range of values for i and j that are less than the degree 
        # (allowing them to be the same size as the degree)
        for i in range(0, degree+1):
            for j in range(0, degree+1):
                #this condition allows us to check that we have our triangular shape.
                #This condition comes from (2.6) in the lectures notes.
                if i + j <= degree:
                    points.append(np.array([j/degree, i/degree]))
        return np.array(points)


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`
    
    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    if not grad:
        if cell.dim == 1:
            x = points[:, 0]
            van_mat = []
            for power in range(0, degree+1):
                van_mat.append(np.power(x, power))
            return np.array(van_mat).T
        if cell.dim == 2:
            x = points[:, 0]
            y = points[:, 1]
            van_mat = []
            for curr_deg in range(0, degree+1):
                for n in range(0, curr_deg+1):
                    curr_x_power = np.power(x, curr_deg-n)
                    curr_y_power = np.power(y, n)
                    van_mat.append(np.multiply(curr_x_power, curr_y_power))
            return np.array(van_mat).T
    else:
        if cell.dim ==1 :
            x = points[:, 0]
            van_grad = []
            for power in range(0, degree+1):
                if power == 0:
                    van_grad.append(np.zeros(x.shape))
                else:
                    van_grad.append(power*np.power(x, power-1))
            return np.expand_dims(np.array(van_grad).T, axis=2)
        if cell.dim == 2:
            x = points[:, 0]
            y = points[:, 1]
            van_mat = []
            for curr_deg in np.arange(0., degree+1):
                for n in np.arange(0, curr_deg+1):

                    partial_x = np.multiply((curr_deg-n)*np.power(x, curr_deg-n-1), np.power(y, n))
                    partial_y = np.multiply(np.power(x, curr_deg-n), n*np.power(y, n-1))

                    if n == 0 and curr_deg == 0:
                        partial_x = np.zeros(x.shape)
                        partial_y = np.zeros(y.shape)
                    elif n==0 and curr_deg != 0:
                        partial_y = np.zeros(y.shape)
                    elif n != 0 and curr_deg ==0:
                        partial_x = np.zeros(y.shape)
                    
                    curr_grad = np.array([partial_x, partial_y])
                    van_mat.append(curr_grad)

            correctly_indexed = np.transpose(np.array(van_mat), (2, 0, 1))
            return np.nan_to_num(correctly_indexed)



class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        #Simply call the ``vandermonde_matrix`` function we have created with the element's own
        #``self.cell``, ``self.degree`` and ``self.nodes`` parameters that have been passed
        #through at the object's initialisation. Then take the inverse of this matrix, and
        #store this into ``self.basis_coefs``
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(self.cell, self.degree, self.nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        if not grad:
            return vandermonde_matrix(self.cell, self.degree, points) @ self.basis_coefs
        else:
            return np.einsum("ijk,jl->ilk",
            vandermonde_matrix(self.cell, self.degree, points, grad=True),
            self.basis_coefs)

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        return np.array([fn(x) for x in self.nodes])

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """
        nodes = lagrange_points(cell, degree)
        def make_entity(self, cell, degree, nodes):
            """Given a specific known ordering of the lagrange elements, builds the entity list
            Although this code looks un-readable, its mostly due to the nested nature of all of the
            lists, and trying to assign the right nodes to each entity."""
            if cell.dim == 1:
                vertex_dict = {0: [0], 1:[degree]}
                edge_dict = {0:[i for i in range(1, degree)]}
                return {0:vertex_dict, 1:edge_dict}
            if cell.dim == 2:
                if degree == 1:
                    return {0:{0:[0], 1:[1], 2:[2]},
                            1:{0:[], 1:[], 2:[]},
                            2:{0:[]}}
                else:
                    vertex_dict = {0:[0], 1:[degree], 2:[len(nodes)-1]}
                    edge_dict = {2:[i for i in range(1, degree)]}
                    edge_0 = [2*degree]
                    for i in range(degree-1, 1, -1):
                        edge_0.append(edge_0[-1]+i)
                    edge_1 = [degree+1]
                    for i in range(degree, 2, -1):
                        edge_1.append(edge_1[-1]+i)
                    edge_dict[0] = edge_0
                    edge_dict[1] = edge_1

                    face_nodes =[]
                    for i in range(degree+2, len(nodes)-1):
                        if (i not in edge_0) and (i not in edge_1) and (i not in 
                        [0, degree, len(nodes)-1]):
                            face_nodes.append(i)

                    face_dict = {0:face_nodes}

                    entities = {0:vertex_dict, 1:edge_dict, 2:face_dict}
                    return entities                
        entities = make_entity(self, cell, degree, nodes)
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes=entities)

class VectorFiniteElement(FiniteElement):
    def __init__(self, fe):
        self.cell = fe.cell
        self.degree = fe.degree
        self.fe = fe
        self.dim = len(fe.entity_nodes)
        self.entity_nodes = fe.entity_nodes

        for i in len(range(self.entity_nodes.keys())):
            for j in len(range(self.entity_nodes[i].keys())):
                self.entity_nodes[i][j] = tuple([self.dim*self.entity_nodes[i][j]+i for i in range(self.dim-1)])

        self.noders_per_entity = self.dim*fe.nodes_per_entity
    def tabulate(self, points, grad=False):
        temp_tabulate = self.fe.tabulate(points, grad=grad)
        if not grad:
            e = np.array([[1,0], [0,1]])
            to_return = []
            for i in range(2*temp_tabulate.shape[0]+1):
                    to_return.append(np.array([temp_tabulate[i//2, i//2], temp_tabulate[1, i//2]]))
            


