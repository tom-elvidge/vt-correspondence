import numpy as np
import nibabel as nib
from skimage import measure
from stl import mesh
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
import glob


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


def init_visited(volume, border):
    # Fill border with ones so that we don't have to check bounds.
    visited = np.zeros(volume.shape)
    visited[0:border, :, :] = 1
    visited[volume.shape[0]-(border+1):volume.shape[0]-1, :, :] = 1
    visited[:, 0:border, :] = 1
    visited[:, volume.shape[1]-(border+1):volume.shape[1]-1, :] = 1
    visited[:, :, 0:border] = 1
    visited[:, :, volume.shape[2]-(border+1):volume.shape[2]-1] = 1
    return visited


def extend_mask(mask):
    # Scan entire mask and extended anything masked.
    extended_mask = np.copy(mask)
    visited = init_visited(mask, 1)
    i = 0
    while i < mask.shape[0]:
        j = 0
        while j < mask.shape[1]:
            k = 0
            while k < mask.shape[2]:
                point = (i, j, k)
                # Point is masked and unvisited.
                if mask[point] == 0 and visited[point] == 0:
                    for neighbour in get_neighbours(point):
                        extended_mask[neighbour] = 0  # Mask neighbours.
                    visited[point] = 1  # Mark original point as visited.
                k += 1
            j += 1
        i += 1
    return extended_mask


def get_neighbours(point):
    x = point[0]
    y = point[1]
    z = point[2]
    return ((x-1, y, z), (x+1, y, z), (x, y-1, z), (x, y+1, z), (x, y, z-1), (x, y, z+1))


def match_file_paths(mask_file_paths, mri_file_paths):
    matches = []
    for mri_file_path in mri_file_paths:
        # Expecting name to be something like "S1_neutralVT_preprocessed".
        name = mri_file_path.split("\\")[-1].split(".")[0]
        # Look for mask file path containing name.
        mask_file_path_match = None
        for mask_file_path in mask_file_paths:
            if name in mask_file_path:
                mask_file_path_match = mask_file_path
        # Add to matches.
        if mask_file_path_match != None:
            matches.append((mri_file_path, mask_file_path_match))
    return matches


def merge_mask(filepaths):
    # Extract filepaths.
    preprocessed_filepath = filepaths[0]
    mask_filepath = filepaths[1]

    # Open preprocessed MRI.
    nifti_file = nib.load(preprocessed_filepath)
    preprocessed = nifti_file.get_fdata()

    # Open mask.
    nifti_file = nib.load(mask_filepath)
    mask = nifti_file.get_fdata()

    # Extended the mask by one voxel, 8 times.
    e_mask = np.copy(mask)
    i = 0
    while i < 8:
        e_mask = extend_mask(e_mask)
        i += 1

    # Hardcoded region to merge. Should work for entire dataset.
    # merge_position = (8, 530, 412)
    merge_position = (8, 470, 512)
    merge_size = (146, 140, 200)
    # merge_position = (4, 240, 206)
    # merge_size = (73, 95, 120)

    # Copy the extended mask onto the preprocessed MRI.
    merge(preprocessed, e_mask, merge_position, merge_size)

    # Save the merged MRI.
    # Expecting name to be something like "S1_neutralVT_preprocessed".
    preprocessed_name = preprocessed_filepath.split("\\")[-1].split(".")[0]
    filename = preprocessed_name + "_mask_merged.nii"
    nib.save(nib.Nifti1Image(preprocessed, np.eye(4)),
             "gen_surface_models\\merged_masks\\{}".format(filename))


if __name__ == "__main__":
    # Check osseous_masks directory for the osseous masks.
    mask_file_paths = glob.glob("gen_surface_models\\osseous_masks\\*.nii")
    # Check preprocessed directory for the preprocessed MRIs.
    mri_file_paths = glob.glob("gen_surface_models\\preprocessed\\*.nii")
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
            pool.map(merge_mask, matches)
    else:
        print("Exiting. Unnessercary masks should be deleted from gen_surface_models\\osseous_masks.")
