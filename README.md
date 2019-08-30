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

### Environment variables:

the shell script setup_classifier.sh contains export commands that set the following environment variables:

* `AZURE_SECRET`: full path to file holding Azure secret information
* `CSCIHUB_SECRET`: full path to file holding Copernicus SciHub secret information
* `PROCESS_DIR`
* `SEN2COR_BIN` : e.g. `/home/tothepoles/PycharmProjects/Sen2Cor-02.05.05-Linux64/bin/`

The two "secret" variables are paths to files containing the account keys for the azure blob storage and Copernicus Sentinelsat APIs. The secret files should be included in the .gitignore file so that they are not added to the Git repository.
 
Create a file to hold Azure secret information, e.g. `.azure_secret`, with format:

    [account]
    user=
    key=

And for Copernicus SciHub:
    
    [account]
    user=
    password=

Create your `PROCESS_DIR`, e.g. `/scratch/BigSurfClass/`. The PROCESS_DIR is the folder where temporary files, images and eventually the output data will be stored. The pickled classifier and mask should also be saved to the PROCESS_DIR in advance of running the classifier script.

To set these environment variables, open a terminal and use `source setup_classifier.sh` to execute whenever it is necessary.

### Template

The user-defined variable values are all defined in a template file (e.g. swgris.template). This is where the main script grabs values to configure the image downloading, processing, classification and reporting.
The user should enter their desired values into the template file prior to running the classifier. Multiple templates can be saved and called along with the classifier.

## Use

There are two main steps: (1) pre-processing Sentinel-2 imagery, and (2) running the classification and albedo algorithms. Driver scripts are provided in this repository to accomplish these steps.

Before you begin, make sure you have created a `template` file containing the settings for your desired workflow, and that you have set the environment variables needed by the workflow (see 'Setup' above).

If you have created the suggestion bash script, then simply run:

    source setup_classifier.sh


### Pre-processing

Run `python download_process_s2.py <template.template>`.


### Classification

Run `python run_classifier.py <template.template>`.


## Contributions

* Joseph Cook (University of Sheffield, University of Aberystwyth)
* Andrew Tedstone (University of Bristol).