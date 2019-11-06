# BigIceSurfClassifier
this repository contains in-development code for automated downloading, processing, analysis and visualizing of Sentinel-2 data for the Greenland Dark Zone over time.


## Setup

### Environment
The environment can be set up manually using the following commands:

    conda create -n IceSurfClassifier -c conda-forge python ipython xarray scikit-learn georaster gdal matplotlib
    conda activate azure
    pip install azure-storage-blob

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

## Use

There are two main steps: (1) pre-processing Sentinel-2 imagery, and (2) running the classification and albedo algorithms. Driver scripts are provided in this repository to accomplish these steps.

Before you begin, make sure you have created a `template` file containing the settings for your desired workflow, and that you have set the environment variables needed by the workflow (see 'Setup' above).

If you have created the suggested bash shell script, then simply run:

    source setup_classifier.sh

### Snicar retrievals

There is an option in the template file to retrieve snicar parameters. If this is set to "True" then the spectral reflectance in each pixel of the S2 tile is compared to a lookup table of
snicar-generated spectra. The snicar parameters (grain size, density, dust concentration, algae concentration) used to generate the closest-matching spectra are assigned to that pixel, producing maps of ice physical properties and light absorbing particle concentrations. 

Note that despite the LUT approach, the snicar retrieval is computatonally expensive and would ideally be run on some HPC resource. We are using a Microsoft Azure D64_v3s Linux Data Science Machine with 64 cores to distribute the processing, which enables the retrieval function to complete in 53 minutes per tile. Testing on JC's laptop (i7-7700 GHz processor, 8 cores, 32GB RAM)took more than 10 hours to retrieve the snicar parameters for a single Sentinel-2 tile. Increasing the size of the LUT increases the computation time significantly. Currently the LUT comprises 2058 individual snicar runs, produced by running snicar with all possible combinations of 6 grain sizes, 7 densities, 7 dust concentrations and 7 algal concentrations. 

### Pre-processing

Run `python download_process_s2.py <template.template>`.


### Classification

Run `python run_classifier.py <template.template>`.


## Contributions

* Joseph Cook (University of Sheffield, University of Aberystwyth)
* Andrew Tedstone (University of Bristol).