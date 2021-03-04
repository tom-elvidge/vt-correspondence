import numpy as np
import nibabel as nib
from skimage import measure
from stl import mesh
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
import glob
from osseous_mask import mask_region_growing, _init_queued, _get_neighbours
from skimage.measure import marching_cubes
from constants import SEEDS, OSSEOUS_ROIS, THRESHOLD


def merge(mri, mask, merge_position, merge_size):
    # Copy mask onto mri for the positions defined.
    i = merge_position[0]
    while i <= merge_size[0] + merge_position[0]:
        j = merge_position[1]
        while j <= merge_size[1] + merge_position[1]:
            k = merge_position[2]
            while k <= merge_size[2] + merge_position[2]:
                if mask[i, j, k] == 1:
                    mri[i, j, k] = 1
                # Leave points where mask is 0 as is.
                k += 1
            j += 1
        i += 1


def match_file_paths(mask_file_paths, mri_file_paths):
    matches = []
    for mri_file_path in mri_file_paths:
        # Expecting name to be something like "S1_neutralVT_preprocessed".
        name = mri_file_path.split("/")[-1].split(".")[0]
        # Look for mask file path containing name.
        mask_file_path_match = None
        for mask_file_path in mask_file_paths:
            if name in mask_file_path:
                mask_file_path_match = mask_file_path
        # Add to matches.
        if mask_file_path_match != None:
            matches.append((mri_file_path, mask_file_path_match))
    return matches


def save_surface(faces, verts, filename):
    solid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            solid.vectors[i][j] = verts[f[j], :]
    solid.save(filename)


def extend_mask(mask, roi_start, roi_end):
    # Scan roi and extended anything masked.
    extended_mask = np.copy(mask)
    visited = _init_queued(mask)
    i = roi_start[0]
    while i < roi_end[0]:
        j = roi_start[1]
        while j < roi_end[1]:
            k = roi_start[2]
            while k < roi_end[2]:
                point = (i, j, k)
                # Point is masked and unvisited.
                if mask[point] == 0 and visited[point] == 0:
                    for neighbour in _get_neighbours(point):
                        extended_mask[neighbour] = 0  # Mask neighbours.
                    visited[point] = 1  # Mark original point as visited.
                k += 1
            j += 1
        i += 1
    return extended_mask


def generate_isosurface(filepaths):
    # Extract filepaths.
    preprocessed_filepath = filepaths[0]
    mask_filepath = filepaths[1]

    # Open preprocessed MRI.
    nifti_file = nib.load(preprocessed_filepath)
    preprocessed = nifti_file.get_fdata()

    # Open mask.
    nifti_file = nib.load(mask_filepath)
    mask = nifti_file.get_fdata()

    # Restrict mask to ROI mask.
    seed = SEEDS[preprocessed_filepath.split("/")[-1]]
    roi = OSSEOUS_ROIS[preprocessed_filepath.split("/")[-1]]
    start = np.asarray(roi["start"])
    end = np.asarray(roi["end"])
    roi_mask = np.zeros(mask.shape)
    roi_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]
             ] = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    # Extended the mask by 10 voxels.
    i = 0
    while i < 10:
        roi_mask = extend_mask(roi_mask, start, end)
        i += 1

    # Add mask on top of preprocessed image.
    prepro_mask = preprocessed + roi_mask
    prepro_mask[prepro_mask > 1] = 1  # Treshold values to 1.

    # Expecting name to be something like "S1_neutralVT_preprocessed".
    preprocessed_name = preprocessed_filepath.split("/")[-1].split(".")[0]
    # Save preprocessed and mask merged.
    # filename = preprocessed_name + "_mask_merged.nii"
    # nib.save(nib.Nifti1Image(prepro_mask, np.eye(4)),
    #          "gen_surface_models/merged_masks/{}".format(filename))

    # Generate final mask.
    final_mask = mask_region_growing(seed, prepro_mask, THRESHOLD)
    # Save mask
    filename = preprocessed_name + "_final_mask.nii"
    nib.save(nib.Nifti1Image(final_mask, np.eye(4)),
             "gen_surface_models/masks/{}".format(filename))
    # Generate isosurface using same threshold.
    verts, faces, normals, values = marching_cubes(final_mask, THRESHOLD)
    # Save as an stl file.
    filename = preprocessed_name + ".stl"
    save_surface(
        faces, verts,  "gen_surface_models/surfaces/{}".format(filename))


if __name__ == "__main__":
    # Check osseous_masks directory for the osseous masks.
    mask_file_paths = glob.glob("gen_surface_models/masks/*_mask_*.nii")
    # Check preprocessed directory for the preprocessed MRIs.
    mri_file_paths = glob.glob("gen_surface_models/preprocessed/*.nii")
    # Match the masks and preprocessed mris.
    matches = match_file_paths(mask_file_paths, mri_file_paths)
    matches_strings = [" - ".join(match) for match in matches]
    print("Found mask and preprocssed MRI matches:\n{}".format(
        "\n".join(matches_strings)))

    # Wait for user to verify matches before starting the merges.
    correct_matches = input("Correct matches? (Y/n) ") != "n"
    if correct_matches:
        # Merge each osseous mask with it's corresponding preprocessed MRI.
        with Pool() as pool:
            pool.map(generate_isosurface, matches)
    else:
        print("Exiting. Unnessercary masks should be deleted from gen_surface_models/osseous_masks.")
