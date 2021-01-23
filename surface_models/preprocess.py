import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, median_filter
import glob
from multiprocessing import Pool
import sys

def preprocess(file_path):
    # Redirect stdout to a logfile.
    sys.stdout = open("temp/preprocess.log","w")

    # Open MRI.
    print("Opening {}...".format(file_path))
    nifti_file = nib.load(file_path)
    mri = nifti_file.get_fdata()

    # Linear interpolation (order=1) in all axis to double size (8-fold???).
    #print("Scaling...")
    #mri = zoom(mri, (2, 2, 2), order=1)

    # Smooth with a median filter to preserve edges.
    print("Applying median filter...")
    mri = median_filter(mri, size=5)

    # Clip and normalise voxels.
    print("Normalising...")
    voxelvals = mri.ravel()
    minval = np.percentile(voxelvals, 5)
    maxval = max(voxelvals)
    # Clip 5% lowest intensities to remove noise in air.
    mri = np.clip(mri, minval, maxval)
    # Normalise.
    mri = ((mri - minval) / (maxval - minval))

    # Save as a new nifti file.
    output_filepath = "surface_models/preprocessed/" + file_path.split("\\")[1].split(".")[0] + "_preprocessed.nii"
    nib.save(nib.Nifti1Image(mri, np.eye(4)), output_filepath)
    print("Saved preprocessed to {}.\n".format(output_filepath))

if __name__ == "__main__":
    # Check data directory for MRIs.
    file_paths = glob.glob("data/*.nii.gz")
    print("Found MRIs:\n{}\n".format("\n".join(file_paths)))

    # Preprocess each MRI.
    with Pool() as pool:
        pool.map(preprocess, file_paths)
