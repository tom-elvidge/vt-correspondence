import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, median_filter
import glob

# Check data directory for MRIs.
mri_paths = glob.glob("data/*.nii.gz")
print("Found MRIs:\n{}\n".format("\n".join(mri_paths)))

# Preprocess each MRI.
for mri_path in mri_paths:
    # Open MRI.
    print("Opening {}...".format(mri_path))
    nifti_file = nib.load("data/S1_contrast1.nii.gz")
    mri = nifti_file.get_fdata()

    # Linear interpolation (order=1) in all axis to double size (8-fold???).
    print("Scaling...")
    mri = zoom(mri, (2, 2, 2), order=1)

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
    preprocessed_path = "surface_models/preprocessed/" + mri_path.split("\\")[1].split(".")[0] + "_preprocessed.nii"
    nib.save(nib.Nifti1Image(mri, np.eye(4)), preprocessed_path)
    print("Saved preprocessed MRI to {}.\n".format(preprocessed_path))
