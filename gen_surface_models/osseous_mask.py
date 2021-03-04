import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import glob
import sys
from collections import deque
from multiprocessing import Pool
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from constants import SEEDS, THRESHOLD


def _get_neighbours(point):
    """
    Get all the immediate neighbours of a 3D point.

    Parameters:
        point (tuple): Tuple of length 3, describing a 3D point.

    Returns:
        tuple: Tuple of tuples, each describing a neighbouring point of the passed point.
    """
    # Pull coords out of point.
    x = point[0]
    y = point[1]
    z = point[2]
    return ((x-1, y, z), (x+1, y, z), (x, y-1, z), (x, y+1, z), (x, y, z-1), (x, y, z+1))


def _init_queued(volume):
    """
    Initialise a matrix giving logical 1 if the corresponding index in volume has been queued. Fill edges of queued
    with ones so that we don't have to check bounds for neighbours.
    """
    queued = np.zeros(volume.shape)
    queued[0, :, :] = 1
    queued[volume.shape[0]-1, :, :] = 1
    queued[:, 0, :] = 1
    queued[:, volume.shape[1]-1, :] = 1
    queued[:, :, 0] = 1
    queued[:, :, volume.shape[2]-1] = 1
    return queued


def mask_region_growing(seed, volume, threshold, max_iterations=sys.maxsize):
    """
    Create a mask of volume using the passed seed and threshold via iterative growing.

    Parameters:
        seed (tuple): Tuple of length 3, describing a 3D point where to start the algorithm.
        volume (numpy.double): The volumetric image to create a mask using.
        threshold (double): The threshold for voxel values to be included in the mask.
        max_iterations (int): The number of iterations before stopping.

    Returns:
        numpy.double: The mask created as described.
    """
    # Generate mask via iterative region growing in 3D.
    queued = _init_queued(volume)
    mask = np.ones(volume.shape)
    q = deque([seed])
    i = 0
    while q and i < max_iterations:  # While q not empty.
        # Get next point from the front of q.
        point = q.popleft()
        # Update point in mask.
        if volume[point] < threshold:
            mask[point] = 0
            # Only add neighbours to q if current voxel in mask.
            # Add all neighbours to q (if not already queued).
            for neighbour in _get_neighbours(point):
                if queued[neighbour] == 0:
                    q.append(neighbour)
                    queued[neighbour] = 1
        # Keep track of iterations
        i += 1
    # Remove border of 1s.
    mask[0, :, :] = 0
    mask[mask.shape[0]-1, :, :] = 0
    mask[:, 0, :] = 0
    mask[:, mask.shape[1]-1, :] = 0
    mask[:, :, 0] = 0
    mask[:, :, mask.shape[2]-1] = 0
    # Return final mask.
    return mask


def calc_surface_area(faces, verts):
    """
    Compute the surface area of the surface described by the passed faces and vertices.

    Parameters:
        verts (numpy.double): The vertices of the surface (N*3 dimensions).
        faces (numpy.int): The faces of the surface described by referencing verts (F*3 dimensions).

    Returns:
        double: The surface area of the surface with the passed faces and vertices.
    """
    # Calculate the surface area of a mesh from it's triangle faces.
    # faces: List of all the faces on the surface. Each face indexes three
    #        points from verts which make up the triangle face.
    # verts: List of all the vertices on the surface.
    area = 0
    for face in faces:
        # Extract x's and y's from the face's vertices.
        xs = [verts[face[0]][0], verts[face[1]][0], verts[face[2]][0]]
        ys = [verts[face[0]][1], verts[face[1]][1], verts[face[2]][1]]
        # Compute area of face from triangle points.
        base = max(xs) - min(xs)
        height = max(ys) - min(ys)
        area += 0.5 * (base + height)
    return area


def osseous_mask(file_path, surface_area_plot=True, iterations=15):
    # Open MRI.
    print("\nOpening {}...".format(file_path))
    nifti_file = nib.load(file_path)
    mri = nifti_file.get_fdata()

    # Get seed for this file from seeds.
    seed = SEEDS[file_path.split("/")[-1]]

    # Generate progressively smoother masks.
    i = 1
    surface_areas = []
    while i < iterations:
        print("\nSmoothing iteration {}.".format(i))

        # Apply Gaussian filter.
        print("Applying Gaussian filter...")
        # Greater sigma than paper since it was taking too long.
        mri = gaussian_filter(mri, sigma=2)

        # Generate mask.
        print("Generating mask via iterative region growing...")
        # Hardcoded threshold from experimentation.
        # Limit the iterations, large enough to fill the vocal tract but not the air around the head.
        mask = mask_region_growing(seed, mri, THRESHOLD, 10000000)

        # Save mask as a new nifti file.
        path_base = "gen_surface_models/masks/" + \
            file_path.split("/")[2].split(".")[0]
        mask_path = path_base + "_mask_{}.nii".format(i)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        print("Saved mask to {}.".format(mask_path))

        # Create surface.
        verts, faces, normals, values = marching_cubes(mask, THRESHOLD)
        # Compute surface area.
        surface_areas.append(calc_surface_area(faces, verts))

        # Next iteration.
        i += 1

    if surface_area_plot:
        # Save figure of surface area per iteration.
        plt.plot(range(1, len(surface_areas)+1), surface_areas)
        plt.xlabel("Iterations")
        plt.ylabel("Surface Area")
        plt.savefig(path_base + "_surface_areas.png")


if __name__ == "__main__":
    # Check preprocessed directory for preprocessed MRIs.
    file_paths = glob.glob("gen_surface_models/preprocessed/*.nii")
    print("Found preprocessed MRIs:\n{}".format("\n".join(file_paths)))

    # Generate a osseous mask for each preprocessed MRI.
    with Pool() as pool:
        pool.map(osseous_mask, file_paths)
