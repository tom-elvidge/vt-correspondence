import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import glob
import sys
from collections import deque
from multiprocessing import Pool

def get_midsagittal_index(volume):
    # Get the halfway index of the first dimension.
    return int(volume.shape[0]/2)

def get_neighbours(point):
    # Pull coords out of point.
    x = point[0]
    y = point[1]
    z = point[2]
    return ((x-1, y, z), (x+1, y, z), (x, y-1, z), (x, y+1, z),(x, y, z-1), (x, y, z+1))

def init_queued(volume):
    # Fill edges of queued with ones so that we don't have to check bounds.
    queued = np.zeros(volume.shape)
    queued[0,:,:] = 1
    queued[volume.shape[0]-1,:,:] = 1
    queued[:,0,:] = 1
    queued[:,volume.shape[1]-1,:] = 1
    queued[:,:,0] = 1
    queued[:,:,volume.shape[2]-1] = 1
    return queued

def mask_region_growing(seed, volume, threshold, max_iterations=sys.maxsize):
    # Generate mask via iterative region growing in 3D.
    queued = init_queued(volume)
    mask = np.ones(volume.shape)
    q = deque([seed])
    i = 0
    while q and i < max_iterations: # While q not empty.
        # Get next point from the front of q.
        point = q.popleft()
        # Update point in mask.
        if volume[point] < threshold:
            mask[point] = 0
        # Add all neighbours to q (if not already queued).
        for neighbour in get_neighbours(point):
            if queued[neighbour] == 0:
                q.append(neighbour)
                queued[neighbour] = 1
        # Keep track of iterations
        i += 1
    return mask

def calc_surface_area(faces, verts):
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

def osseous_mask(file_path):
    # Open MRI.
    print("\nOpening {}...".format(file_path))
    nifti_file = nib.load(file_path)
    mri = nifti_file.get_fdata()

    # Hardcoded seed. Centre works for almost the entire dataset.
    # seed = (81, 513, 513)
    seed = (40, 255, 255)

    # Generate 8 progressively smoother masks. Will manually select the most appropiate.
    i = 1
    max_iterations = 6
    while i < max_iterations:
        print("\nSmoothing iteration {}.".format(i))

        # Apply Gaussian filter.
        print("Applying Gaussian filter...")
        #mri = gaussian_filter(mri, sigma=4) # Greater sigma than paper since it was taking too long.
        mri = gaussian_filter(mri, sigma=2) 

        # Generate mask.
        print("Generating mask via iterative region growing...")
        # Hardcoded threshold from experimentation.
        # Limit the iterations, large enough to fill the vocal tract but not the air around the head.
        mask = mask_region_growing(seed, mri, 0.1, 10000000)

        # Save mask as a new nifti file.
        mask_path = "surface_models\\osseous_masks\\" + file_path.split("\\")[2].split(".")[0] + "_mask_{}.nii".format(i)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        print("Saved mask to {}.".format(mask_path))

        # Next iteration.
        i += 1


if __name__ == "__main__":
    # Check preprocessed directory for preprocessed MRIs.
    file_paths = glob.glob("surface_models\\preprocessed\\*.nii")
    print("Found preprocessed MRIs:\n{}".format("\n".join(file_paths)))

    # Generate a osseous mask for each preprocessed MRI.
    with Pool() as pool:
        pool.map(osseous_mask, file_paths)