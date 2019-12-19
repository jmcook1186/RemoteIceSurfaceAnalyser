# Big Ice Surf Classifier

This repository contains in-development code for automated downloading, processing, analysis and visualizing of Sentinel-2 data for the Greenland Dark Zone. For a user defined tile and date range, the script will download the imagery, reproject and apply an atmospheric correction, and then for each pixel predict a discrete surface type using a random forest classifier trained on fiels spectroscopy data, invert a radiative transfer model to retrieve ice grain size, density and light absorbing impurity concentrations, and calculate the surface albedo. Maps for each parameter are saved as jpgs and the summary data are saved as csv files for each tile/date.

## Setup

### Hardware
The computational requirements of this script vary depending upon which functions are toggled on/off. The script is generally suitable for running on a powerful laptop/desktop as we have been tried to keep as much out of memory as possible, only loading arrays into memory when realy necessary. However, the invert_snicar() function is demanding and can take up to a day to run on my 8 core i7-7700 32GB RAM laptop. If using the invert_snicar() function an HPC resource is recommended. I have been running the full pipeline on a 64 core Azure Linux Data Science Machine, in which case one tile takes 53 mins to process. With the SNICAR inversion and the EB model functions both toggled ON, ~1 hour 5 mins per tile. Bear in mind that in Nov 2019, the VM cost is ~£3/hour.

### Environment
The environment can be set up manually using the following commands:

    conda create -n IceSurfClassifier -c conda-forge python ipython xarray scikit-learn gdal georaster gdal seaborn rasterio matplotlib
    pip install azure sklearn-xarray sentinelsat dask[complete]

or alternatively the environment can be built from environment.yaml using:

    conda env -f environment.yml

In this case the environment name is set in the first line of the yml file. Beware of the prefix in the final line of the file - this will need to be updated or deleted depending on the installation.

### Environment variables

The classifier requires several environment variables to be set:

* `AZURE_SECRET`: full path to file holding Azure secret information
* `CSCIHUB_SECRET`: full path to file holding Copernicus SciHub secret information
* `PROCESS_DIR`
* `SEN2COR_BIN` : e.g. `~/Sen2Cor-02.05.05-Linux64/bin/`

The two "secret" variables are paths to files containing the account keys for the azure blob storage and Copernicus Sentinelsat APIs. These files can be located anywhere on your system and preferably outside the repository (although the `.gitignore` file is set to ignore `.secret` files to reduce the risk of these files being committed accidentally).
 
Create a file to hold Azure secret information, e.g. `.azure_secret`, with format:

    [account]
    user=
    key=

And for Copernicus SciHub:
    
    [account]
    user=
    password=

Create your `PROCESS_DIR`, e.g. `/scratch/BigSurfClass/`. The `PROCESS_DIR` is the folder where temporary files, images and eventually the output data are stored. The pickled classifier and mask should also be saved to the `PROCESS_DIR` in advance of running the classifier script.

The simplest way to set these environment variables is to use a shell script. An example, `setup_classifier.sh`, has been provided with this repository. Make a copy that you can modify to suit your environment. The copy should not be committed back to this repository.

With a shell script created to set these environment variables, execute `source setup_classifier.sh` in the terminal in which you plan to run the classifier. The variables will be available for the duration of the session. 

### Template

The user-defined variable values are all defined in a template file (e.g.  `swgris.template`). This is where the main script grabs values to configure the image downloading, processing, classification and reporting.

The user should enter their desired values into the template file prior to running the classifier. Multiple templates can be saved and called along with the classifier.

Note that the LUT parameters are included in the template file. These MUST be identical to the parameters used to generate the LUT used in the snicar inversion, else the retrieval will grab the wrong values. Unless the user has generated a new LUT, this should NOT be changed.


### Directory Structure

The directories should be arranged as follows. It is critical that this structure is maintained else paths will fail.

```
Big Ice Surf Classifier
|
|----- run_classifier.py
|----- Big_Sentinel_Classifier.py
|----- BISC_param_Log.txt
|----- BISC_plot_figures.py
|----- download_process_s2.py
|----- ebmodel.py
|----- environment.yml
|----- README.md
|----- sentinel2_azure.py
|----- sentinel2_tools.py
|----- setup_classifier.sh
|----- xr_cf_conventions.py
|----- swgris.template
|----- .azure_secret
|----- .cscihub_secret
|
|
|-----Process_Dir
|           |
|           |---ICE_MASK.nc
|           |---merged_mask.tif
|           |---Sentinel2_classifier.pkl
|           |---SNICAR_LUT_2058.npy
|           |---will be populated with band images and auto-flushed at end of run
|           |
|            Outputs
|               |
|               |
|              Tile
|               |
|               |---To be populated with output data
|
|
```


## How to run

There are two main steps: (1) pre-processing Sentinel-2 imagery, and (2) running the classification, albedo, snicar retrieval and eb-modelling algorithms. These are configured by altering the template file.

Before you begin, make sure you have created a `template` file containing the settings for your desired workflow, and that you have set the environment variables needed by the workflow (see 'Setup' above).

If you have created the suggested bash shell script, then simply run:

    source setup_classifier.sh

The configuration details will automatically be saved to a text file (BISC_Param_Log.txt). A list of the tile/date combinations that are discarded due to QC flags or download errors are saved to csv files (aborted_downloads.csv, rejected_by_qc.csv) and so is a list of all tile/date combinations successfully analysed (good_tiles.csv).


### Pre-processing

Run `python download_process_s2.py <template.template>`.


### Processing

Run `python run_classifier.py <template.template>`.


## Functionality

### Classification

The classifier assigns one of 6 surface type labels to each pixel. This is achieved using a random forest classifier trained on field spectroscopy data and validated using multispectral imagery acquired from a UAV flown over the surface of the Greenland Ice Sheet in summer 2017 and 2018. 

### Albedo

Surface albedo is calculated using Liang et al's (2000) narrowband to broadband conversion which was originally formulated for LANDSAT but validated for Sentinel-2 by Naegeli et al. (2017).

### Snicar retrievals

There is an option in the template file to retrieve snicar parameters. If this is set to "True" then the spectral reflectance in each pixel of the S2 tile is compared to a lookup table of snicar-generated spectra. The snicar parameters (grain size, density, dust concentration, algae concentration) used to generate the closest-matching spectra are assigned to that pixel, producing maps of ice physical properties and light absorbing particle concentrations. 

Note that despite the LUT approach, the snicar retrieval is computatonally expensive and would ideally be run on some HPC resource. We are using a Microsoft Azure D64_v3s Linux Data Science Machine with 64 cores to distribute the processing, which enables the retrieval function to complete in 53 minutes per tile. Testing on a single S2 tile on JC's laptop (i7-7700 GHz processor, 8 cores, 32GB RAM) was aborted after 10 hours. Increasing the size of the LUT increases the computation time significantly. Currently the LUT comprises 2058 individual snicar runs, produced by running snicar with all possible combinations of 6 grain sizes, 7 densities, 7 dust concentrations and 7 algal concentrations. 

### Missing pixel interpolation
Images where pixels have been masked out using the CloudMask (the sensivity of which is controlled by the user-defined variable cloudCoverThreshold) can have those pixels infilled using linear interpolation by toggling the "interpolate_cloud" option in the template. This simple function is applied to the dataarrays before they are masked and collated into the dataset that is then saved to netcdf as a model output.

### Missing date interpolation

There is an option to toggle interpolation on/off. If interpolation is toggled on, the script will use pixel-wise linear interpolation to generate synthetic datasets for each missing date in the time series. Dates may be missing from the time series sue to a lack of overpass or a data quality issue such as excessive cloud cover. The interpolation function identifies these missing tiles, then identifies the most recent and nearest future dates where a valid dataset was acquired. It then applies a point-to-point linear regression between the values in each pixel in the past image and the corresponding pixel in the future image. The pixel values are then predicted using the regression equation for the missing dates. Recommend also scanning through the images manually after interpolation because occasionally interpolating pixels that were cloud-free in the past and cloudy in the future or vice versa can lead to unrealistic interpolation results. Anomalous dates should be manually discarded or alternatively the interpoaltion function could be run as a standalone with "good" past and future tiles manually selected.

### Energy Balance Modelling

There is an option to toggle energy balance modelling on/off. If this is toggled on, the albedo calculated using the Liang et al. (2000) formula is used as an input to drive a Python implementation of the Brock and Arnold (2000) point-surface energy balance model. By default, all available processing cores are used to distribute this task using the multiprocessing package. Currently, the other meteorological variabes are hard-coded constants - an obvious to-do is to make these variables that are grabbed from a LUT for the specific day/tile being processed. This is more computatonally expensive than the classification and albedo computations but cheaper than the snicar parameter retrievals. It is feasible to run the processing pipeline with energy-balance toggled on locally on a well spec'd laptop in about 100 minutes per tile. For large runs it is recommended to use an HPC resource to accelerate this function (takes about 12 mins on 64 core VM), especially if both the energy balance and snicar param retrievals are toggled on (in which case ~70 mins per tile). The outputs from this function are melt rate in mm w.e./day. The total melt over the tile is also returned.

## Troubleshooting

Here are some of the possible exceptions, errors and gotchas that could be encountered when running this code, and their solutions.

### Azure and SentinelHub
The user must have their own Azure account with blob storage access. The scripts here manage the creation of storage blobs and i/o to and from them, but this relies upon the user having a valid account name and access key saved in an external file ".azure_secret" in the process directory. The same is true of the copernicus science hub - the user must have a valid username and password stored in an external file ".cscihub_secret" saved in the process directory to enable batch downloads of Sentinel-2 images that are not already saved in blob storage.

###   File "/...", line x, in load_img_to_xr, raise ValueError
This error likely results from having surplus .jp2 images in the process directory. This can happen when a previous run was aborted before the clear_process_dir() function was run, or when the user has added files accidentally, because that can cause multiple files to be returned by the glob in the load_img_to_xr() function which should only return unique filenames. To resolve, clear the .jp2 files from the process directory and retry.

### Black backgrounds in saved figures
This code probably shouldn't be run in an interactive (e.g. ipython/jupyter) environment anyway, but in case it is used in this way it is worth noting that running the code in a jupyer interactive session in VSCode with the dark theme enabled will cause the saved figures to have black backgrounds, as the "dark theme" is persisted in preference to any mpl configurations set in the code. It is easy to avoid this by running the BISC_plot_figures.py script directly from the terminal (recommended), or using a different IDE (PyCharm works).

### permission error in netcdf save
This likely results from a path error, check that the folder structure on the machine running the code is consistent with that in this README.

### ... config.read_file(open(sys.argv[1])) IndexError: list index out of range
This error most likely indicates that run_classifier.py is being run from the terminal wihout specifying the .template file or that there is a problem with the .template file. Perhaps the template file name has been typed incorrectly.


## Development notes

To test a full run without downloading files from blob, comment out lines 196 - 202 in sentinel2_azure.py. This prevents the call to download the files from blob and will simply run the function on the files already present in the process_dir. Obviously this only works for single tiles and will not iterate over multiple tiles. Might be a good idea to start a devlopemnt version that retrieves tiles locally instead of from blob storage to help speed up testing.

Dec 19th 2019: VM disk space being used up. Changed workflow so that .nc files are uploaded to blob storage and deleted from local storage on creation. Deactivated summary statistics and dataset concatenation functions with the intention of calling these separately post-hoc, i.e. populate blob storage using VM then later download, concatenate and analyse datasets locally to save expensive compute time on VM. Unsurprisingly introducing uploads to blob storage has slowed down the script somewhat. Expect approximately 1.5 hours per tile for the complete sequence including EB modelling and snicar inversion function.

Dec 19th 2019: added zipped folder of PROMICE AWS data for three sites to the repo. Intention is to use these data to provide input data to the eb model.


## Contributions

### Authors
* Joseph Cook (University of Aberystwyth)
* Andrew Tedstone (University of Bristol)
