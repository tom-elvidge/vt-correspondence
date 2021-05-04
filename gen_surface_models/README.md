Inside this directory is all the source code for generating surface models of the vocal tract from MRIs.

My method is based loosely off the method described in [this paper](https://www.scitepress.org/Papers/2017/61383).

Place the raw volumetric MRI files in the data/mri directory.

Install the requirements.

```bash
prbx$ pip install -r requirements.txt
```

Preprocess the MRIs.

```bash
prbx$ python gen_surface_models/preprocess.py
```

They will be saved to gen_surface_models/preprocessed.

Update constants.py with the seed position and osseous mask ROI for the preprocessed MRI, an example is in the file.

Compute all the osseous masks for each preprocessed MRI. Defaults to 15 iterations but this can be changed. This will use up a lot of disk space so I advise doing these one at a time.

```bash
prbx$ python gen_surface_models/osseous_mask.py
```

Each mask will be saved to gen_surface_models/masks as well as a surface area plot at each iteration. Use this to select the best osseous mask and delete all others.

Generate the final segmentation mask and surface mesh.

```bash
prbx$ python gen_surface_models/generate_surface.py
```

This will save the final segmentation mask to gen_surface_models/masks and the surface mesh to gen_surface_models/surfaces.