from point_cloud_tools import transform, new_transformation, flattern_vectors, make_vectors
import numpy as np
from stl import mesh
from svdt import svdt
import random
import math


class RANSAC_SVD:
    """
    RANdomly SAmpled Consensus with Singular Value Decomposition for rigid transformations. Estimate the rigid
    transformation between two point clouds using SVD but adding RANSAC to give resistance to outliers.

    Attributes:
        _iterations (int):
            Number of iterations to consider a new random sample.

        _error_threshold (float):
            Threshold for the root mean squared error between corresponding points to vote for a transformation.

        _consensus_threshold (float):
            Proportion of points required to vote for the transformation to consider it as a new best.

        _sample_size (int):
            Number of points to randomly sample to compute a transformation.

    Methods:
        set_error_threshold(error_threshold):
            Setter for _error_threshold.

        set_consensus_threshold(consensus_threshold):
            Setter for _consensus_threshold.

        set_iterations(iterations):
            Setter for _iterations.

        set_sample_size(sample_size):
            Setter for _sample_size.

        run(source, target, return_rmse=False, debug=False):
            Returns the best transformation from source to target after running RANSAC for _iterations.
    """

    def __init__(self, iterations=100, error_threshold=0, consensus_threshold=0.5, sample_size=3):
        # iterations = log(1-p)/log(1-e^s) where p: probability of success, e: outlier ratio, s: sample size.
        self._iterations = iterations
        self._error_threshold = error_threshold
        self._consensus_threshold = consensus_threshold
        self._sample_size = sample_size

    def set_error_threshold(self, error_threshold):
        self._error_threshold = error_threshold

    def set_consensus_threshold(self, consensus_threshold):
        self._consensus_threshold = consensus_threshold

    def set_iterations(self, iterations):
        self._iterations = iterations

    def set_sample_size(self, sample_size):
        self._sample_size = sample_size

    def run(self, source, target, return_rmse=False, debug=False):
        """
        Returns the best transformation from source to target after running RANSAC for the specified
        number of iterations. Matching indexes in source and target should correspond, this is used
        to evaluate the transformation.

        Parameters:
            source (numpy.double):
                The point cloud to transform.

            target (numpy.double):
                The point cloud to try and transform source close to.

            return_rmse (bool):
                If True then it also returns the rmse of the best transformation.

            debug (bool):
                If True then it prints debug messages while running.

        Returns:
            best_transformation (numpy.double):
                The best transformation from source to target.

            best_rmse (numpy.double):
                Only returned if return_rmse passed as True.
                The root mean squared error between the transformed source and target.
        """
        # Keep track of the current best transformation and it's associated rmse.
        # Initially the identity matrix since this does no transformation.
        best_transformation = np.identity(source.shape[1]+1)
        # Initally the best_rmse is very high.
        best_rmse = 9999

        # Run RANSAC _iterations times.
        j = 0
        while j < self._iterations:
            if debug:
                print("RANSAC iteration {}".format(j))

            # Get _sample_size many random indicies from source.
            # Index i in source should correspond to index i in target.
            maybe_inliers = random.sample(
                range(0, source.shape[0]), self._sample_size)
            # Create a list of outlier indicies.
            maybe_outliers = list(range(0, source.shape[0]))
            for i in maybe_inliers:
                maybe_outliers.remove(i)

            # Get best rotation and translation between randomly sampled points from source and target.
            R, L, rmse = svdt(source[maybe_inliers],
                              target[maybe_inliers], order="row")
            tr = new_transformation(R, L)

            # Apply transformation to source point cloud.
            source_tr = transform(source, tr)

            # Consensus voting of maybe_outliers.
            consensus_inliers = []  # Points from maybe_outliers which agree with transformation
            for i in maybe_outliers:
                # Compute rmse between transforced source point and corresponding target point.
                diff = source_tr[i] - target[i]
                err = math.sqrt(np.dot(diff, diff.T))
                # Error must be less than threshold for this point to vote for the transformation.
                if err <= self._error_threshold:
                    # if debug:
                    #     print("Vote for transformation ({} <= {}).".format(
                    #         err, self._error_threshold))
                    consensus_inliers.append(i)

            # If enough from maybe_outliers vote for the transformation.
            consensus = (len(consensus_inliers)/len(maybe_outliers))
            if consensus >= self._consensus_threshold:
                if debug:
                    print("Consider for new best transformation ({} >= {}).".format(
                        consensus, self._consensus_threshold))
                # Consider this transformation for new best_transformation.
                all_inliers = maybe_inliers + consensus_inliers
                R, L, rmse = svdt(source[all_inliers],
                                  target[all_inliers], order="row")
                # If better rmse then update best_transformation and best_rmse.
                if rmse < best_rmse:
                    if debug:
                        print("New best transformation found with rmse ({} < {}).".format(
                            rmse, best_rmse))
                    best_transformation = new_transformation(R, L)
                    best_rmse = rmse

            # Try another transformation.
            j += 1

        # If the best_rmse was never updated set it to None here.
        if best_rmse == 9999:
            best_rmse = None

        # Return best_rmse if set by user.
        if return_rmse:
            return best_transformation, best_rmse
        # Otherwise just return best_transformation.
        return best_transformation
