Place surface meshes in data/surface_models with the file name '{name}_final.stl'.

Place segmented hard palate surface meshes in data/surface_models with the file name '{name}_final_hp.stl'.

Place hard palate picked points from MeshLab in data/surface_models with the file name '{name}.pp'.

The name must be the same for each that need to be matched e.g. 's1_schwa_final.stl', 's1_schwa.pp', etc...

Install the requirements.

```bash
prbx$ pip install -r requirements.txt
```

Run the registration script.

```bash
prbx$ python registration/registration.py
```

Confirm the file matches.

```bash
Parameters for register function:
['data/surface_models/s1_ah_final.stl', 'data/surface_models/s1_ah_final_hp.stl', 'data/surface_models/s1_ah.pp', 'data/surface_models/s1_schwa_final.stl', 'data/surface_models/s1_schwa_final_hp.stl', 'data/surface_models/s1_schwa.pp']
...
Correct parameters? (Y/n) 
```

The transformed surface mesh will be saved in registration/surface_meshes.
