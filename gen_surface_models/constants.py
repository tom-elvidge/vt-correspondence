THRESHOLD = 0.1

SEEDS = {
    "S1_neutralVT_preprocessed.nii": (80, 540, 540),
    "S1_contrast1_preprocessed.nii": (80, 540, 540),
}

OSSEOUS_ROIS = {
    "S1_neutralVT_preprocessed.nii": {
        "start": (10, 490, 450),
        "end": (150, 680, 600)
    },
    "S1_contrast1_preprocessed.nii": {
        "start": (5, 490, 330),
        "end": (155, 680, 590)
    }
}
