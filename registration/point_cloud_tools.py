import numpy as np
import matplotlib.pyplot as plt


def new_transformation(R, L):
    """ 
    Combines a rotation matrix and translation vector into a single transformation matrix.

    Parameters: 
        R (numpy.double): Rotation matrix.
        L (numpy.double): Translation vector.

    Returns: 
        numpy.double: Rotation and translation transformation matrix.
    """
    tr = np.identity(R.shape[0] + 1)
    # Add rotation.
    i = 0
    while i < R.shape[0]:
        j = 0
        while j < R.shape[1]:
            tr[j, i] = R[i, j]
            j += 1
        i += 1
    # Add translation.
    i = 0
    while i < L.shape[0]:
        tr[R.shape[0], i] = L[i]
        i += 1
    return tr


def calc_rmse(points1, points2):
    """ 
    Calculates the root mean squared error between two point clouds. Point clouds must
    the same dimensions and corresponding indicies.

    Parameters: 
        points1 (numpy.double): 
        points1 (numpy.double): 

    Returns: 
        double: Root mean squared error between points1 and points2.
    """
    errors = points1 - points2
    root_squared_errors = np.sqrt(np.multiply(errors, errors))
    rmse = np.sum(root_squared_errors) / len(root_squared_errors)
    return rmse


def transform(points, tr):
    """ 
    Applies the transform to the point cloud.

    Parameters: 
        points (numpy.double): Point cloud to transform.
        tr (numpy.double): Transformation matrix to apply to point cloud.

    Returns: 
        numpy.double: The translated point cloud.
    """
    # Add extra column of 1's to allow dot product for transfo.
    points_tr = np.ones((points.shape[0], points.shape[1]+1))
    points_tr[:, :-1] = points
    # Apply transformation.
    points_tr = np.dot(points_tr, tr)
    # Remove extra column of 1's.
    points_tr = np.delete(points_tr, np.s_[3], axis=1)
    return points_tr


def flattern_vectors(vectors):
    """ 
    Flattern matrix of all vectors into matrix of vertices and a new matrix of vectors which indexes vertices.

    Allows the use of the vertices as point clouds while still maintaining the surface information for the stl file.
    The reverse of this function make_vectors(vertices, vectors_ind) recombines them into vectors, allowing it to be
    saved as an stl file.

    Parameters: 
        vectors (numpy.double): NxMxD matrix, N vectors, M points in each vector, D dimensions in each point.

    Returns: 
        (numpy.double, numpy.double): KxD matrix of all vertices, and NxM matrix of vectors which indexes vertices.
    """
    # Flatterned matrix of all vertices.
    vertices = np.zeros(
        (vectors.shape[0] * vectors.shape[1], vectors.shape[2]))
    # Matrix of all vectors with indexes to vertices.
    vectors_ind = np.zeros((vectors.shape[0], vectors.shape[1]))
    # Flatterned vector index.
    vi = 0
    # For each vector.
    i = 0
    while i < vectors.shape[0]:
        # For each vertex in vector.
        j = 0
        while j < vectors.shape[1]:
            # Get each vertex from vectors.
            vertices[vi, :] = vectors[i, j, :]
            # Index the vertices for each vector.
            vectors_ind[i, j] = vi
            # Next vertex.
            j += 1
            vi += 1
        # Next vector.
        i += 1
    return vertices, vectors_ind


def make_vectors(vertices, vectors_ind):
    """ 
    Combine the vertices and vectors_ind into a single matrix where the vertices are in each vector rather than being indexed.

    Parameters: 
        vertices (numpy.double): KxD matrix of all vertices.
        vectors_ind (numpy.double): NxM matrix of vectors which indexes vertices.

    Returns: 
        numpy.double: NxMxD matrix, N vectors, M points in each vector, D dimensions in each point.
    """
    # New vectors matrix which combines vertices and vectors_ind.
    vectors = np.zeros(
        (vectors_ind.shape[0], vectors_ind.shape[1], vertices.shape[1]))
    # For each vector.
    i = 0
    while i < vectors_ind.shape[0]:
        # For each vertex.
        j = 0
        while j < vectors_ind.shape[1]:
            # Combine vertices and vectors_ind.
            vectors[i][j] = vertices[int(vectors_ind[i, j]), :]
            j += 1
        i += 1
    return vectors
