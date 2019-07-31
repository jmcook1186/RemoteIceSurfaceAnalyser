# BigIceSurfClassifier
this repository contains in-development code for automated downloading, processing, analysis and visualizing of Sentinel-2 data for the Greenland Dark Zone over time.


## Setup

conda environment:

    conda create -n IceSurfClassifiers python=3.6 numpy matplotlib scikit-learn seaborn azure rasterio gdal pandas
    conda install -c conda-forge xarray georaster sklearn_xarray

or (AJT current test environment before I saw the statement I've now pasted in above):

    conda create -n azure -c conda-forge python ipython xarray scikit-learn georaster gdal matplotlib
    conda activate azure
    pip install azure-storage-blob


Environment variables:

* `AZURE_SECRET`: full path to file holding Azure secret information
* `CSCIHUB_SECRET`: full path to file holding Copernicus SciHub secret information
* `PROCESS_DIR`
* `SEN2COR_BIN` : e.g. `/home/tothepoles/PycharmProjects/Sen2Cor-02.05.05-Linux64/bin/`

Create a file to hold Azure secret information, e.g. `.azure_secret`, with format:

    [account]
    user=
    key=

And for Copernicus SciHub:
    
    [account]
    user=
    password=

Create your `PROCESS_DIR`, e.g. `/scratch/BigSurfClass/`.

You may want to set these environment variables in a shell file which you then `source` to execute whenever it is necessary. An example is provided in this repository.


## Contributions

* Joseph Cook (University of Sheffield, University of Aberystwyth)
* Andrew Tedstone (University of Bristol).