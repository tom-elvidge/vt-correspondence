import numpy as np
import nibabel as nib
from skimage import measure
from stl import mesh
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
import glob
from osseous_mask import mask_region_growing
from skimage.measure import marching_cubes


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


def extend_mask(mask, roi_start, roi_end):
    # Scan entire mask and extended anything masked.
    extended_mask = np.copy(mask)
    visited = init_visited(mask, 1)
    i = roi_start[0]
    while i < roi_end[0]:
        j = roi_start[1]
        while j < roi_end[1]:
            k = roi_start[2]
            while k < roi_end[2]:
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


def save_surface(faces, verts, filename):
    solid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            solid.vectors[i][j] = verts[f[j], :]
    solid.save(filename)


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
    ini = np.asarray((10, 490, 450))  # start
    s = np.asarray((140, 190, 150))  # size
    e = ini + s  # end
    roi_mask = np.zeros(mask.shape)
    roi_mask[ini[0]:e[0], ini[1]:e[1], ini[2]:e[2]
             ] = mask[ini[0]:e[0], ini[1]:e[1], ini[2]:e[2]]

    # Extended the mask by 10 voxels.
    i = 0
    while i < 10:
        roi_mask = extend_mask(roi_mask, ini, e)
        i += 1

    # Add mask on top of preprocessed image.
    prepro_mask = preprocessed + roi_mask
    prepro_mask[prepro_mask > 1] = 1  # Treshold values to 1.

    # Expecting name to be something like "S1_neutralVT_preprocessed".
    preprocessed_name = preprocessed_filepath.split("/")[-1].split(".")[0]
    filename = preprocessed_name + "_mask_merged.nii"
    nib.save(nib.Nifti1Image(prepro_mask, np.eye(4)),
             "gen_surface_models/merged_masks/{}".format(filename))

    # Generate final mask.
    final_mask = mask_region_growing((80, 540, 540), prepro_mask, 0.1)
    # Remove border from mask.
    final_mask = remove_border(final_mask)
    filename = preprocessed_name + "_final_mask.nii"
    nib.save(nib.Nifti1Image(final_mask, np.eye(4)),
             "gen_surface_models/merged_masks/{}".format(filename))
    # Generate isosurface using same threshold.
    verts, faces, normals, values = marching_cubes(final_mask, 0.1)
    # Save as an stl file.
    filename = preprocessed_name + "_surface.stl"
    save_surface(
        faces, verts,  "gen_surface_models/merged_masks/{}".format(filename))


def remove_border(mask):
    mask[0, :, :] = 0
    mask[mask.shape[0]-1, :, :] = 0
    mask[:, 0, :] = 0
    mask[:, mask.shape[1]-1, :] = 0
    mask[:, :, 0] = 0
    mask[:, :, mask.shape[2]-1] = 0
    return mask


if __name__ == "__main__":
    # Check osseous_masks directory for the osseous masks.
    mask_file_paths = glob.glob("gen_surface_models/osseous_masks/*.nii")
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
