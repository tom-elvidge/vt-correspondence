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
from landmarks import meshlab_picked_points_to_dict, correspond_picked_points
from point_cloud_tools import transform, new_transformation, flattern_vectors, make_vectors
import sys
from multiprocessing import Pool
import glob
from transformation_solver import RANSAC_SVD, SVD


def plot_3d(points):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="red")
    plt.show()


def plot_3d_2(pointsa, pointsb):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointsa[:, 0], pointsa[:, 1], pointsa[:, 2], c="red")
    ax.scatter(pointsb[:, 0], pointsb[:, 1], pointsb[:, 2], c="blue")
    plt.show()


def register(source_filename, source_hp_filename, source_pp, target_filename, target_hp_filename, target_pp):
    # Process output filename.
    output_name = source_filename.split("/")[-1].split(".")[0]
    output_dir = "registration/surface_models/"

    # Load the hard palate mesh for source.
    source_hp = mesh.Mesh.from_file(source_hp_filename)
    source_hp_verts, source_hp_vectors = flattern_vectors(source_hp.vectors)

    # Load the hard palate mesh for target.
    target_hp = mesh.Mesh.from_file(target_hp_filename)
    target_hp_verts, target_hp_vectors = flattern_vectors(target_hp.vectors)

    # Load the full mesh for source.
    source = mesh.Mesh.from_file(source_filename)
    source_verts, source_vectors = flattern_vectors(source.vectors)

    # Load the full mesh for target.
    target = mesh.Mesh.from_file(target_filename)
    target_verts, target_vectors = flattern_vectors(target.vectors)

    # Get manual landmarks from picked points file.
    source_landmarks = meshlab_picked_points_to_dict(source_pp)
    target_landmarks = meshlab_picked_points_to_dict(target_pp)
    # Reorder landmarks to correspond.
    landmarks = correspond_picked_points([source_landmarks, target_landmarks])
    source_landmarks = landmarks[0]
    target_landmarks = landmarks[1]

    # Get tranformation that registers hard palates.
    ransac_svd = RANSAC_SVD(error_threshold=5)
    icp = ICP(iterations=5, transformation_solver=ransac_svd)
    tr, rmses = icp.run(source_hp_verts, target_hp_verts,
                        source_landmarks, target_landmarks, return_landmark_rmses=True, debug=True)

    # # Plot original hard palates.
    # plot_3d_2(s3_c1_hp_verts, s3_n_hp_verts)
    # # Plot transformed sourcehard palate against target.
    # plot_3d_2(transform(source_hp_verts, tr), s3_n_hp_verts)

    # Plot landmark_rmses.
    plt.plot(range(len(rmses)), rmses)
    plt.xlabel("Iterations")
    plt.ylabel("Landmark RMSE")
    # Save plot as an image.
    plt.savefig(output_dir + output_name + "_pp_rmse_per_iter.png")

    # Transform full source.
    source_verts_tr = transform(source_verts, tr)

    # # Plot original.
    # plot_3d_2(source_verts, target_verts)
    # # Plot transformed against target.
    # plot_3d_2(source_verts_tr, target_verts)

    # Save the transformed full source as new stl file.
    source_vectors_tr = make_vectors(source_verts_tr, source_vectors)
    source.vectors = source_vectors_tr
    source.save(output_dir + output_name + "_tr.stl")


if __name__ == "__main__":
    # subject/articulations to register.
    registration_pairs = [
        ("s1_contrast1", "s1_neutralV"),
        ("s2_contrast1", "s2_neutralV"),
        ("s3_contrast1", "s3_neutralV")
    ]

    # Get all file paths.
    full_stl_files = glob.glob("data/surface_models/*_final.stl")
    hp_stl_files = glob.glob("data/surface_models/*_final_hp.stl")
    landmark_files = glob.glob("data/surface_models/*.pp")

    # Get parameters for register function.
    register_parameters = []
    for pair in registration_pairs:
        params = [None, None, None, None, None, None]
        # Full stl files.
        for filename in full_stl_files:
            # source_filename
            if pair[0] in filename:
                params[0] = filename
            # target_filename
            if pair[1] in filename:
                params[3] = filename
        # Hard palate stl files.
        for filename in hp_stl_files:
            # source_hp_filename
            if pair[0] in filename:
                params[1] = filename
            # target_hp_filename
            if pair[1] in filename:
                params[4] = filename
        # Picked points files.
        for filename in landmark_files:
            # source_pp
            if pair[0] in filename:
                params[2] = filename
            # target_pp
            if pair[1] in filename:
                params[5] = filename
        # Add to register_parameters.
        register_parameters.append(params)

    # Output all params.
    print("Parameters for register function:")
    for params in register_parameters:
        print(params)
    # Wait for user input before calling register function.
    correct_params = input("Correct parameters? (Y/n) ") != "n"
    if correct_params:
        # Call register with each list of parameters.
        with Pool() as pool:
            pool.starmap(register, register_parameters)
    else:
        print(
            "Exiting. Ensure all full stl, hard palate stl and picked points files exist.")
