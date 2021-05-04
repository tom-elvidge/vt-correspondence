from icp import ICP
import numpy as np
from stl import mesh
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from svdt import svdt
import math
import random
from landmarks import import_meshlab_pp_file, correspond_picked_points
from point_cloud_tools import transform, new_transformation, flattern_vectors, make_vectors
import sys
from multiprocessing import Pool
import glob
from transformation_solver import RANSAC_SVD, SVD
from matplotlib.lines import Line2D


def svdt_test():
    # Manually defined point cloud
    A = np.matrix([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # Known rotation and translation.
    R = np.matrix([
        [math.cos(20), -math.sin(20), 0],
        [math.sin(20), math.cos(20), 0],
        [0, 0, 1]
    ])
    print(R)
    L = np.asarray([1, 1, 1])
    print(L)
    T = new_transformation(R, L)
    # Transform point cloud by known transformation.
    B = transform(A, T)
    # Plot the two point clouds.
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], c='red')
    ax.scatter3D(B[:, 0], B[:, 1], B[:, 2], c='blue')
    plt.savefig('temp/before.png')
    # Find rotation and translation which transforms the original point cloud to the one with a known transformation.
    R, L, rmse = svdt(A, B, order="row")
    # Output the rotation and translation to compare.
    print(R)
    print(L)
    # Apply found transformation.
    A = transform(A, new_transformation(R, L))
    # Re plot the point clouds to show they are aligned.
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], c='red')
    ax.scatter3D(B[:, 0], B[:, 1], B[:, 2], c='blue')
    plt.savefig('temp/after.png')


def icp_test():
    # Manually defined point cloud
    A = np.matrix([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # Known rotation and translation.
    R = np.matrix([
        [math.cos(20), -math.sin(20), 0],
        [math.sin(20), math.cos(20), 0],
        [0, 0, 1]
    ])
    print(R)
    L = np.asarray([1, 1, 1])
    print(L)
    T = new_transformation(R, L)
    # Transform point cloud by known transformation.
    B = transform(A, T)
    # Plot the two point clouds.
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], c='red')
    ax.scatter3D(B[:, 0], B[:, 1], B[:, 2], c='blue')
    plt.savefig('temp/before.png')
    # Find rotation and translation via ICP.
    tr, landmark_rmses, closest_point_rmses = ICP().run(
        A, B, s_landmarks=A, t_landmarks=B,  return_rmses=True, debug=True)
    print(closest_point_rmses)
    print(landmark_rmses)
    print(tr)
    # Apply found transformation.
    A = transform(A, tr)
    # Re plot the point clouds to show they are aligned.
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], c='red')
    ax.scatter3D(B[:, 0], B[:, 1], B[:, 2], c='blue')
    plt.savefig('temp/after.png')


def ransac_error_threshold():
    # Plot consensus'.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.set_ylabel('Consensus Ratio', fontsize=16)
    ax.set_xlabel('Error Threshold (t)', fontsize=16)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)

    colors = {
        's1': 'red',
        's2': 'blue',
        's3': 'green'
    }

    ts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    for subject in ['s1', 's2', 's3']:
        # Load the hard palate mesh for source.
        source_hp = mesh.Mesh.from_file(
            f'data/surface_models/{subject}_contrast1_final_hp.stl')
        source_hp_verts, source_hp_vectors = flattern_vectors(
            source_hp.vectors)

        # Load the hard palate mesh for target.
        target_hp = mesh.Mesh.from_file(
            f'data/surface_models/{subject}_neutralVT_final_hp.stl')
        target_hp_verts, target_hp_vectors = flattern_vectors(
            target_hp.vectors)

        # Rearrange for closest points.
        nn = NearestNeighbors(
            n_neighbors=1, algorithm="kd_tree").fit(target_hp_verts)
        target_indicies = nn.kneighbors(
            source_hp_verts, n_neighbors=1, return_distance=False)
        target_hp_verts = target_hp_verts[target_indicies.ravel()]

        # Try different error thresholds.
        consensus_all = []
        for t in ts:
            solver = RANSAC_SVD(iterations=35, error_threshold=t,
                                consensus_threshold=0.5, sample_size=3)
            tr, rmse, consensus = solver.run(
                source_hp_verts, target_hp_verts, return_rmse=True, return_consensus=True, debug=True)
            consensus_all.append(consensus)

        # Add consensus line for this subject.
        ax.plot(np.mean(np.matrix(consensus_all), axis=1),
                color=colors[subject], label=subject)

    ax.legend(prop={"size": 16})

    plt.savefig('temp/consensusplot.png')


def icp_rmse_within_subject():
    # results = {
    #     'Subject 1': {
    #         'Landmarks': [1,1,1,1]
    #         'Closest Points': [1,1,1,1]
    #     }
    # }
    results = {}
    # Do ICP for each subject and add the results to dict.
    for subject in ['s1', 's2', 's3']:
        # Load the hard palate mesh for source and target. Registering contrast1 to neutralVT.
        source_hp = mesh.Mesh.from_file(
            f'data/surface_models/{subject}_contrast1_final_hp.stl')
        s, source_hp_vectors = flattern_vectors(source_hp.vectors)
        target_hp = mesh.Mesh.from_file(
            f'data/surface_models/{subject}_neutralVT_final_hp.stl')
        t, target_hp_vectors = flattern_vectors(target_hp.vectors)

        # Get manual landmarks from picked points file.
        source_landmarks = import_meshlab_pp_file(
            f'data/surface_models/{subject}_contrast1_final_picked_points.pp')
        target_landmarks = import_meshlab_pp_file(
            f'data/surface_models/{subject}_neutralVT_final_picked_points.pp')
        # Reorder landmarks to correspond.
        landmarks = correspond_picked_points(
            [source_landmarks, target_landmarks])
        s_lm = landmarks[0]
        t_lm = landmarks[1]

        # Compute transformation which registers source to target.
        solver = SVD()
        icp = ICP(iterations=5, transformation_solver=solver)
        tr, lm_rmses, cp_rmses = icp.run(
            s, t, s_lm, t_lm, return_rmses=True, debug=False)

        # Add to results.
        subject_results = {}
        subject_results['Landmarks'] = lm_rmses
        subject_results['Closest Points'] = cp_rmses
        results[subject] = subject_results

    # Return and plot later.
    return results


def icp_ransac_rmse_within_subject():
    # results = {
    #     'Subject 1': {
    #         'Landmarks': [1,1,1,1]
    #         'Closest Points': [1,1,1,1]
    #     }
    # }
    results = {}
    # Do ICP for each subject and add the results to dict.
    for subject in ['s1', 's2', 's3']:
        # Load the hard palate mesh for source and target. Registering contrast1 to neutralVT.
        source_hp = mesh.Mesh.from_file(
            f'data/surface_models/{subject}_contrast1_final_hp.stl')
        s, source_hp_vectors = flattern_vectors(source_hp.vectors)
        target_hp = mesh.Mesh.from_file(
            f'data/surface_models/{subject}_neutralVT_final_hp.stl')
        t, target_hp_vectors = flattern_vectors(target_hp.vectors)

        # Get manual landmarks from picked points file.
        source_landmarks = import_meshlab_pp_file(
            f'data/surface_models/{subject}_contrast1_final_picked_points.pp')
        target_landmarks = import_meshlab_pp_file(
            f'data/surface_models/{subject}_neutralVT_final_picked_points.pp')
        # Reorder landmarks to correspond.
        landmarks = correspond_picked_points(
            [source_landmarks, target_landmarks])
        s_lm = landmarks[0]
        t_lm = landmarks[1]

        # Compute transformation which registers source to target.
        solver = RANSAC_SVD(iterations=35, error_threshold=5,
                            consensus_threshold=0.5, sample_size=3)
        icp = ICP(iterations=5, transformation_solver=solver)
        tr, lm_rmses, cp_rmses = icp.run(
            s, t, s_lm, t_lm, return_rmses=True, debug=False)

        # Add to results.
        subject_results = {}
        subject_results['Landmarks'] = lm_rmses
        subject_results['Closest Points'] = cp_rmses
        results[subject] = subject_results

    # Return and plot later.
    return results


def plot_icp_rmse_within_subject(results, filename):
    # Create some colors and styles.
    colors = {
        's1': 'red',
        's2': 'blue',
        's3': 'green'
    }

    # Create figure.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.set_xlabel("Iterations", fontsize=20)
    ax.set_ylabel("RMSE", fontsize=20)
    ax.set_ylim([0, 10])

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    # Plot each line from results.
    for subject in results.keys():
        plt.plot(results[subject]['Closest Points'],
                 color=colors[subject], label=subject)

    # Add legend.
    ax.legend(prop={"size": 20})

    # Save.
    plt.savefig(f'temp/{filename}')


# plot_icp_rmse_within_subject(
#     icp_rmse_within_subject(), 'icp_rmse_within_subject.png')

# plot_icp_rmse_within_subject(
#     icp_ransac_rmse_within_subject(), 'icp_ransac_rmse_within_subject.png')

ransac_error_threshold()
