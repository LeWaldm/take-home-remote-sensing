# take-home-remote-sensing

This repository contains code for the solution of a take-home task in remote-sensing (Fri. 12th Jan 2024).

Design choices:
- In `Observation_dataset`, samples are indexed with a unique index over all observations. When retrieving a sample, the index is compared to the index of the first sample in each observation to determine the corresponding observation. This is implemented as comparison with a torch tensor, which is parallelized and thus sufficiently fast. 
- Each sample within an observation is identified by its upper left pixel. 
- The mask extracted from the annotations currently only supports exactly one class.

Resources
- working with `.geojson` and polygons: rasterio docs (https://rasterio.readthedocs.io/en/stable/quickstart.html)
- opening satellite image: https://stackoverflow.com/questions/37722139/load-a-tiff-stack-in-a-numpy-array-with-python
- loading json from remote location: https://stackoverflow.com/questions/12965203/how-to-get-json-from-webpage-into-python-script
