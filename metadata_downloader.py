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
from collections import OrderedDict
import pytz
import datetime
from pysolar import *
import pandas as pd


def download_L1C(api, L1Cpath, tile, dates, cloudcoverthreshold):
    """
    This function uses the sentinelsat API to download L1C products for tiles defined by "tile" in the date range
    specified by "dates". Prints number of files and their names to console.
    :param L1Cpath:
    :param tile:
    :param dates:
    :return: L1Cfiles
    """

    # define keyword arguments
    query_kwargs = {
            'platformname': 'Sentinel-2',
            'producttype': 'S2MSI1C',
            'date': dates,
            'cloudcoverpercentage': (0, cloudcoverthreshold)}

    products = OrderedDict()

    # loop through tiles and list all files in date range

    kw = query_kwargs.copy()
    kw['tileid'] = tile  # products after 2017-03-31
    pp = api.query(**kw)
    products.update(pp)

    # keep metadata in pandas dataframe
    out = api.to_dataframe(products)

    return out


def add_coszen(path):

    df = pd.read_csv(path)

    sol_elev = []
    sol_zen = []
    sol_zen_rad=[]
    coszen = []

    for date in df.endposition:
        date = str(date+'+0000')
        dt = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f%z')
        sol_elev.append(get_altitude(67.04, -49.49, dt))

    
    for i in np.arange(0,len(sol_elev),1):
        zenith = 90-sol_elev[i]
        zenith_rad = zenith * np.pi/180
        cos_zenith = np.cos(zenith_rad)
        
        sol_zen.append(zenith)
        sol_zen_rad.append(zenith_rad)
        coszen.append(cos_zenith)

    df['sol_elev'] = sol_elev
    df['sol_zen'] = sol_zen
    df['sol_zen_rad'] = sol_zen_rad
    df['coszen'] = coszen

    df.to_csv(path)

    return

# Get project configuration
config = configparser.ConfigParser()
config.read_file(open(sys.argv[1]))


# Open API to Azure blob store
azure_cred = configparser.ConfigParser()
azure_cred.read_file(open(os.environ['AZURE_SECRET']))
azure = sentinel2_azure.AzureAccess(azure_cred.get('account','user'),
                                    azure_cred.get('account','key'))


# Open API to Copernicus SciHub
cscihub_cred = configparser.ConfigParser()
cscihub_cred.read_file(open(os.environ['CSCIHUB_SECRET']))
chub_api = SentinelAPI(cscihub_cred.get('account','user'), cscihub_cred.get('account','password'),'https://scihub.copernicus.eu/apihub')


metadata = []
tiles = json.loads(config.get('options','tiles'))
years = json.loads(config.get('options','years'))
months = json.loads(config.get('options','months'))

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

            out = download_L1C(chub_api, L1Cpath, tile, dates, config.get('thresholds','cloudCoverThresh'))
            
            savepath = '/home/joe/Desktop/BISC_metadata/{}_{}_{}.csv'.format(tile,year,month)
            out.to_csv(savepath)

    # run function to add solar zenith info to metadata csv
    #add_coszen(savepath)