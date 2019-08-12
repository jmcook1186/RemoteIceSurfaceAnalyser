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

sys.path.append("/home/at15963/scripts/IceSurfClassifiers") 

import sentinel2_tools
import sentinel2_azure
import Big_Sentinel_Classifier as bsc
import xr_cf_conventions

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
                    
                    s2xr = bsc.load_img_to_xr(img_path, 
                        config.get('options', 'resolution'),
                        Icemask, 
                        Cloudmask)

                    classified = bsc.classify_image(clf, s2xr, savepath, tile, date, savefigs=True)
                    albedo = bsc.calculate_albedo(s2xr)

                    ## Collate predicted map, albedo map and projection info into xarray dataset
                    mask2 = bsc.combine_masks(s2xr)

                    # 1) Retrieve projection info from S2 datafile and add to netcdf
                    proj_info = xr_cf_conventions.create_grid_mapping(s2xr.Data.attrs['crs'])

                    # 2) Create associated lat/lon coordinates DataArrays using georaster (imports geo metadata without loading img)
                    # see georaster docs at https:/media.readthedocs.org/pdf/georaster/latest/georaster.pdf
                    # find B02 jp2 file
                    fileB2 = glob.glob(str(img_path + '*B02_20m.jp2'))
                    fileB2 = fileB2[0]

                    lon, lat = xr_cf_conventions.create_latlon_da(fileB2, 'x', 'y')

                    # 3) add predicted map array and add metadata
                    predicted = predicted.fillna(0)
                    predicted = predicted.where(mask2 > 0)
                    predicted.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
                    predicted.name = 'Surface Class'
                    predicted.attrs['long_name'] = 'Surface classified using Random Forest'
                    predicted.attrs['units'] = 'None'
                    predicted.attrs['key'] = config.get('netcdf','predicted_legend')
                    predicted.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                    # add albedo map array and add metadata
                    albedo = albedo.fillna(0)
                    albedo = albedo.where(mask2 > 0)
                    albedo.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
                    albedo.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
                    albedo.attrs['units'] = 'dimensionless'
                    albedo.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                    # collate data arrays into a dataset
                    dataset = xr.Dataset({
                        'classified': (['x', 'y'], predicted),
                        'albedo':(['x','y'],albedo),
                        'Icemask': (['x', 'y'], s2xr.Icemask),
                        'Cloudmask':(['x','y'], s2xr.Cloudmask),
                        'FinalMask':(['x','y'],mask2),
                        proj_info.attrs['grid_mapping_name']: proj_info,
                        'longitude': (['x', 'y'], lon_array),
                        'latitude': (['x', 'y'], lat_array)
                    })

                    dataset = xr_cf_conventions.add_geo_info(dataset, 'x', 'y', 
                        config.get('netcdf','author'), 
                        config.get('netcdf', 'title'))

                    dataset.to_netcdf(savepath + "{}_{}_Classification_and_Albedo_Data.nc".format(tile,date), mode='w')

                    dataset = None

                    if config.get('options', 'savefigs'):
                        cmap1 = mpl.colors.ListedColormap(
                            ['purple', 'white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
                        cmap1.set_under(color='white')  # make sure background is white
                        cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
                        cmap2.set_under(color='white')  # make sure background is white

                        fig, axes = plt.subplots(figsize=(10,8), ncols=1, nrows=2)
                        predictedxr.plot(ax=axes[0], cmap=cmap1, vmin=0, vmax=6)
                        plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
                        plt.title('Greenland Ice Sheet from Sentinel 2 classified using Random Forest Classifier (top) and albedo (bottom)')
                        axes[0].grid(None)
                        axes[0].set_aspect('equal')

                        albedoxr.plot(ax=axes[1], cmap=cmap2, vmin=0, vmax=1)
                        plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
                        axes[1].set_aspect('equal')
                        axes[1].grid(None)
                        fig.tight_layout()
                        plt.savefig(str(savepath + "{}_{}_Classified_Albedo.png".format(tile,date)), dpi=300)
                        plt.close()


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