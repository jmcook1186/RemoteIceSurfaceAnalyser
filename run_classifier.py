#!/usr/bin/env python
"""
This script pulls images relating to specific Sentinel-2 tiles on specific dates
 from Azure blob storage and classifies them using a random forest classifier 
 trained on field spectroscopy data (see github.com/jmcook1186/IceSurfClassifiers).
There is a sequence of quality control functions that determine whether the 
downloaded image is of sufficient quality to be used in the analysis or 
alternatively whether it should be discarded. Reasons for discarding include 
cloud cover, NaNs and insufficient ice relative to land or ocean in the image. 
The sensitivity to these factors is tuned by the user by setting up the template
file.

Usage:
	$ run_classifier.py <template_file.template>

Returns: 
	Nothing

Outputs:
	Files.

Requires these environment variables to be set:
AZURE_SECRET - path and filename of config file containing Azure secret information


"""

import sys
import os
import configparser
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import datetime as dt
import calendar

import sentinel2_tools
import sentinel2_azure
import Big_Sentinel_Classifier as bsc

# Get project configuration
config = configparser.ConfigParser()
config.read_file(open(sys.argv[1]))


# Open API to Azure blob store
azure_cred = configparser.ConfigParser()
azure_cred.read_file(open(os.environ['AZURE_SECRET']))
azure = sentinel2_azure.AzureAccess(azure_cred.get('account','user'),
                                    azure_cred.get('account','key'))


# matplotlib settings: use ggplot style and turn interactive mode off
mpl.style.use('ggplot')
plt.ioff()




###################################################################################
######## DEFINE BLOB ACCESS, GLOBAL VARIABLES AND HARD-CODED PATHS TO FILES #######
###################################################################################

img_path = os.environ['PROCESS_DIR'] 

# set up empty lists and dataframes to append to
download_problem_list =[] # empty list to append details of skipped tiles due to missing info
QC_reject_list = [] # empty list to append details of skipped tiles due to cloud cover
good_tile_list = [] # empty list to append tiles used in analysis
masterDF = pd.DataFrame()
dates = []

###################################################################################
######################### SET TILE AND DATE RANGE #################################
###################################################################################

years = json.loads(config.get('options','years'))
months = json.loads(config.get('options','months'))
# set up dates (this will create list of all days in year/month range specified above)
for year in years:
    for month in months:

        startDate = dt.date(year, month, 1)
        endDate = dt.date(year, month, calendar.monthrange(year,month)[1])
        dates_pd = pd.date_range(startDate, endDate, freq='1D')

        for date in dates_pd:
            dates.append(date.strftime('%Y%m%d'))


###################################################################################
############### RUN FUNCTIONS & HEALTHCHECKS ######################################
###################################################################################

tiles = json.loads(config.get('options','tiles'))
for tile in tiles:

    tile = tile.lower() #azure blob store is case sensitive: force lower case
    #first create directory to save outputs to
    dirName = str(img_path+'/outputs/'+tile+"/")

    # Create target Directory if it doesn't already exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    # make DirName the path to save files to
    savepath = dirName

    for date in dates:
        print("\n *** DOWNLOADING FILES FOR TILE: {} DATE: {} ***\n".format(tile, date))

        # query blob for files in tile and date range
        filtered_bloblist, download_flag = azure.download_imgs_by_date(tile, 
        	date, img_path)

        # check download and only proceed if correct no. of files and cloud layer present
        if download_flag:
            print("\n*** Download Flag Raised ***\n *** Skipping tile {} on {} due to download flag ***".format(tile, date))
            download_problem_list.append('{}_{}'.format(tile,date))

        else:
            print("\n*** No Download flag raised *** \n *** Checking cloud, ice and NaN cover ***")

            Icemask, Cloudmask = sentinel2_tools.format_mask(img_path, 
            	config.get('options','icemask'), 
            	os.environ['PROCESS_DIR'] + '/outputs/ICE_MASK.nc',
            	config.get('thresholds','cloudCoverThresh'))

            QCflag, useable_area = sentinel2_tools.img_quality_control(img_path,
                Icemask, Cloudmask, 
            	config.get('thresholds','minArea'))

            # Check image is not too cloudy. If OK, proceed, if not, skip tile/date
            if QCflag:
                print("\n *** QC Flag Raised *** \n*** Skipping tile {} on {} due to QCflag: {} % useable pixels ***".format(tile, date, np.round(useable_area,4)))
                QC_reject_list.append('{}_{}_useable_area = {} '.format(tile,date,useable_area))

            else:
                print("\n *** No cloud or ice cover flags: proceeding with image analysis for tile {}".format(tile))
                good_tile_list.append('{}_{}_useable_area = {} '.format(tile,date,useable_area))

                try: # use try/except so that any images that slips through QC and then fail do not break run
                    clf = bsc.load_model_and_images(img_path, pickle_path, Icemask, Cloudmask)
                    bsc.ClassifyImages(clf, img_path, savepath, tile, date, savefigs=True)
                    summaryDF, masterDF = bsc.albedo_report(masterDF, tile, date, savepath)

                except:
                    print("\n *** IMAGE ANALYSIS ATTEMPTED AND FAILED FOR {} {}: MOVING ON TO NEXT DATE \n".format(tile,date))

        sentinel2_tools.clear_img_directory(img_path)

    print("\n *** COLLATING INDIVIDUAL TILES INTO FINAL DATASET***")
    concat_dataset = bsc.concat_all_dates(savepath, tile)

    # save logs to csv files
    print("\n *** SAVING QC LOGS TO TXT FILES ***")
    print("\n *** aborted_downloads.txt, rejected_by_qc.txt and good_tiles.txt saved to output folder ")
    np.savetxt(str(savepath+"/aborted_downloads.csv"), download_problem_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath+"/rejected_by_qc.csv"), QC_reject_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath+"/good_tiles.csv"), good_tile_list, delimiter=",", fmt='%s')

print("\n ************************\n *** COMPLETED RUN  *** \n *********************")