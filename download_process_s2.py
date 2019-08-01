#!/usr/bin/env python
"""
The user defines a list of relevant tiles and dates. This script then accesses the SentinelHub store and downloads the
relevant tiles as zipped L1C products and saves them locally. The local file is then converted to the L2A product using
the ESA Sen2Cor processor, overwriting the L1C product. The L2A product is then uploaded to blob storage and then erased
from the local disk.

In this script, two batches of dates are provided for each tile. This is to enable flushing of the local disk before too
many tiles are saved. The files are large, and more than one month's worth of images can lead to the local disk filling
up causing the script to crash. To avoid this, I iterate through month by month instead of providing a large date range.

Blob storage is organised into separate containers for each tile with a separate folder for each date saved inside.

tile
|--date
    |--individual band jp2s


Execution:

Use of this driver script requires an IceSurfClassifiers template file.

    $ download_process_s2.py <myjobfile.template>


Andrew Tedstone, July 2019, based on original script by Joseph Cook

"""

import numpy as np
import os
import sys
import json
import configparser
import datetime as dt
import calendar

from sentinelsat import SentinelAPI

import sentinel2_tools
import sentinel2_azure

# Get project configuration
config = configparser.ConfigParser()
config.read_file(open(sys.argv[1]))

tiles = json.loads(config.get('options','tiles'))
years = json.loads(config.get('options','years'))
months = json.loads(config.get('options','months'))


# Open API to Azure blob store
azure_cred = configparser.ConfigParser()
azure_cred.read_file(open(os.environ['AZURE_SECRET']))
azure = sentinel2_azure.AzureAccess(azure_cred.get('account','user'),
                                    azure_cred.get('account','key'))


# Open API to Copernicus SciHub
cscihub_cred = configparser.ConfigParser()
cscihub_cred.read_file(open(os.environ['CSCIHUB_SECRET']))
chub_api = SentinelAPI(cscihub_cred.get('account','user'), 
        cscihub_cred.get('account','password'),
        'https://scihub.copernicus.eu/apihub')


# Iterate through tiles
for tile in tiles:

    for year in years:

        # Download and pre-process one month at a time
        for month in months:

            # set start and end dates
            startDate = dt.date(year, month, 1)
            endDate = dt.date(year, month, calendar.monthrange(year,month)[1])

            # dates creates a tuple from the user-defined start and end dates
            dates = (startDate, endDate)

            # set path to save downloads
            L1Cpath = os.environ['PROCESS_DIR']

            print('\n TILE %s, %s-%s' %(tile, year, month))

            L1Cfiles = sentinel2_tools.download_L1C(chub_api, L1Cpath, tile, dates, 
                config.get('thresholds','cloudCoverThresh'))
            sentinel2_tools.process_L1C_to_L2A(L1Cpath, L1Cfiles, 
                config.get('options','resolution'), unzip_files=True)
            sentinel2_tools.remove_L1C(L1Cpath)
            upload_status = azure.send_to_blob(tile, L1Cpath, check_blobs=True)
            sentinel2_tools.remove_L2A(L1Cpath, upload_status)

            # Should not catch all exceptions like this as this hides any genuine errors. Instead catch 
            # specific errors only - not yet sure what these would be.
            #except: 
            #    print("\n No images found for this tile on the specified dates")

print('X'.center(80,'X'), ' FINISHED ALL TILES '.center(80,'X'),'\n','X'.center(80,'X'))