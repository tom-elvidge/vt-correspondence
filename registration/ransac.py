from point_cloud_tools import transform, new_transformation, flattern_vectors, make_vectors
import numpy as np
from stl import mesh
from svdt import svdt
import random
import math


def ransac_svdt(source, target, iterations, error_threshold=0, consensus_threshold=0.5, sample_size=3, return_rmse=False, debug=False):
    # error_threshold of 0 only if there are points which are completely rigid i.e. 0 error between them is possible.
    # consensus_threshold is the ratio of points which must agree with the model to consider it further.
    # sample_size is the number of points which are randomly sampled for the initial model.
    # iterations = log(1-p)/log(1-e^s) where p: probability of success, e: outlier ratio, s: sample size.

    best_transformation = np.identity(source.shape[1]+1)
    best_rmse = 999999999
    # Correspond source and target indicies.
    j = 0
    while j < iterations:
        # print("iteration {}".format(j))
        # Get n random indicies from source. Index i in source corresponds to index i in target.
        maybe_inliers = random.sample(range(0, source.shape[0]), sample_size)
        # Create array of outlier indicies.
        maybe_outliers = list(range(0, source.shape[0]))
        for i in maybe_inliers:
            maybe_outliers.remove(i)

        # Solve for rotation and translation that gives best registration between randomly selected points from source and target.
        R, L, rmse = svdt(source[maybe_inliers],
                          target[maybe_inliers], order="row")
        tr = new_transformation(R, L)

        # Transform all source points.
        source_tr = transform(source, tr)
        # Compute Euclidian distance (error metric) between transformed source points and corresponding target points.
        # errors = np.dot(source_tr, target.T)
        # print(errors)

        # Consensus voting of outliers.
        consensus_inliers = []
        # Euclidian distance between corresponding points down main diagonal.
        for i in maybe_outliers:
            # if errors[i, i] < error_threshold:
            # print(np.dot(source_tr[i], target[i].T))
            diff = source_tr[i] - target[i]
            err = np.dot(diff, diff.T)
            # print(err)
            if err <= error_threshold:
                consensus_inliers.append(i)

        # d of points left out agree with transformation.
        agree = (len(consensus_inliers)/len(maybe_outliers))
        if agree >= consensus_threshold:
            if debug:
                print(agree)
            all_inliers = maybe_inliers + consensus_inliers
            # New transformation on all inliers.
            R, L, rmse = svdt(source[all_inliers],
                              target[all_inliers], order="row")
            if rmse < best_rmse:
                if debug:
                    print("new best")
                best_transformation = new_transformation(R, L)
                best_rmse = rmse

        # Next randomly sampled model.
        j += 1
    # Return the best transformation and it's rmse.
    if best_rmse == 999999999:
        best_rmse = None
    if return_rmse:
        return best_transformation, best_rmse
    return best_transformation
