from point_cloud_tools import new_transformation, calc_rmse, transform
from ransac import ransac_svdt
from svdt import svdt
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ICP:

    def __init__(self, iterations=10, closest_point_solver="kd_tree", transformation_solver="svd", debug=False):
        self._iterations = iterations
        self._closest_point_solver = closest_point_solver
        self._transformation_solver = transformation_solver
        self._debug = debug

    def run(self, source, target, s_landmarks=None, t_landmarks=None, return_landmark_rmses=False):
        # Check source and target dimensions.
        if source.shape[1] != target.shape[1]:
            print("source and target must have the same number of dimensions")
            return

        # Check source and target landmark dimensions.
        if s_landmarks.shape[1] != t_landmarks.shape[1]:
            print("s_landmarks and t_landmarks must have the same number of dimensions")
            return

        # Extra variables needed if using ransac.
        if self._transformation_solver == "ransac_svd":
            ransac_error_threshold = 20
            ransac_error_threshold_decay = 0.85

        # Create copies of source and target to not modify originals.
        intermediate_source = np.copy(source)
        intermediate_target = np.copy(target)

        # Identity matrix as initial transformation.
        tr = np.identity(source.shape[1]+1)

        # Record the root mean squared error per iteration.
        landmark_rmses = np.zeros(self._iterations)

        # Start running ICP.
        i = 0
        while i < self._iterations:
            if self._debug:
                print("ICP iteration {}...".format(i))

            # Determine closest point in target for each point in trasnformed.
            if self._closest_point_solver == "kd_tree":
                nn = NearestNeighbors(
                    n_neighbors=1, algorithm="kd_tree").fit(intermediate_target)
            else:
                print("{} is not a supported closest point solver".format(
                    self._closest_point_solver))

            # Reorder the points in intermediate_target such that the points in intermediate_source
            # and intermediate_target with the same index correspond (as closest points).
            targed_indicies = nn.kneighbors(
                intermediate_source, n_neighbors=1, return_distance=False)
            intermediate_target = intermediate_target[targed_indicies.ravel()]

            # Compute transformation that minimises rmse between source and target.
            if self._transformation_solver == "svd":
                R, L, rmse = svdt(intermediate_source,
                                  intermediate_target, order="row")
                # Combine rotation and translation into single transformation matrix for this iteration.
                tr_i = new_transformation(R, L)
            elif self._transformation_solver == "ransac_svd":
                if self._debug:
                    print("RANSAC error threshold: {}".format(
                        ransac_error_threshold))
                tr_i = ransac_svdt(
                    intermediate_source, intermediate_target, 500, error_threshold=ransac_error_threshold, debug=self._debug)
                # Reduce the ransac_error_threshold. More strict with each ICP iteration.
                ransac_error_threshold = ransac_error_threshold * ransac_error_threshold_decay
            else:
                print("{} is not a supported transformation solver".format(
                    self._transformation_solver))

            # Apply current transformation to intermediate_source.
            if self._debug:
                print("Transformation: {}".format(tr_i))
            intermediate_source = transform(intermediate_source, tr_i)

            # Update total transformation.
            tr = np.dot(tr, tr_i)

            # Keep a record of the root mean squared errors.
            if return_landmark_rmses:
                landmark_rmses[i] = calc_rmse(
                    transform(s_landmarks, tr), t_landmarks)

            # Next ICP iteration.
            i += 1

        # Return.
        if return_landmark_rmses:
            return tr, landmark_rmses
        return tr