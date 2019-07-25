#!/usr/bin/env python
"""
Usage:
	run_classifier.py <template_file>

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

import sentinel_azure
import Big_Sentinel_Classifier as bsc


config = configparser.ConfigParser()
config.read_file(open(sys.argv[1]))

azure_cred = configparser.ConfigParser()
azure_cred.read_file(open(os.environ['AZURE_SECRET']))

os.environ['PROCESS_DIR']

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

        # adjust end date for number of days in month
        if month in [1,3,5,7,8,10,12]:
            endDate = 31
        elif month in [4,6,9,11]:
            endDate = 30
        elif month in [2]:
            endDate = 28

        days = np.arange(1,endDate+1,1)

        for day in days:
            date = str(str(year)+str(month).zfill(2)+str(day).zfill(2))
            dates.append(date)


###################################################################################
############### RUN FUNCTIONS & HEALTHCHECKS ######################################
###################################################################################
print(config.get('options','tiles'))
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
        filtered_bloblist, download_flag = sentinel_azure.download_imgs_by_date(tile=tile, 
        	date=date, 
        	img_path=img_path,
        	blob_account_name=azure_access.get('account','user'), 
        	blob_account_key=azure_access.get('account','key'))

        # check download and only proceed if correct no. of files and cloud layer present
        if download_flag:
            print("\n*** Download Flag Raised ***\n *** Skipping tile {} on {} due to download flag ***".format(tile, date))
            download_problem_list.append('{}_{}'.format(tile,date))

        else:
            print("\n*** No Download flag raised *** \n *** Checking cloud, ice and NaN cover ***")

            Icemask, Cloudmask = bsc.format_mask(img_path, 
            	config.get('options','icemask'), 
            	os.environ['PROCESS_DIR'] + '/outputs/ICE_MASK.nc',
            	config.get('thresholds','cloudProb'))

            QCflag, useable_area = bsc.img_quality_control(Icemask, 
            	Cloudmask, 
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

        

        bsc.clear_img_directory(img_path)

    print("\n *** COLLATING INDIVIDUAL TILES INTO FINAL DATASET***")
    concat_dataset = bsc.concat_all_dates(savepath, tile)

    # save logs to csv files
    print("\n *** SAVING QC LOGS TO TXT FILES ***")
    print("\n *** aborted_downloads.txt, rejected_by_qc.txt and good_tiles.txt saved to output folder ")
    np.savetxt(str(savepath+"/aborted_downloads.csv"), download_problem_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath+"/rejected_by_qc.csv"), QC_reject_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath+"/good_tiles.csv"), good_tile_list, delimiter=",", fmt='%s')

print("\n ************************\n *** COMPLETED RUN  *** \n *********************")