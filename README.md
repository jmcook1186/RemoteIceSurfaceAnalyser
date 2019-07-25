# BigIceSurfClassifier
this repository contains in-development code for automated downloading, processing, analysis and visualizing of Sentinel-2 data for the Greenland Dark Zone over time.


## Setup

conda environment:

    conda create -n azure -c conda-forge python ipython xarray scikit-learn georaster gdal matplotlib
    conda activate azure
    pip install azure-storage-blob


Environment variables:

* `AZURE_SECRET`: full path to file holding Azure secret information
* `PROCESS_DIR`

Create a file to hold Azure secret information, e.g. `.azure_secret`, with format:

    [account]
    user=
    key=

