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
import pandas as pd
import json
import datetime as dt
import calendar
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sentinel2_tools
import sentinel2_azure
import Big_Sentinel_Classifier
import xr_cf_conventions
import multiprocessing as mp
import gc

###################################################################################
######## DEFINE BLOB ACCESS, GLOBAL VARIABLES AND SET UP EMPTY LISTS/ARRAYS #######
###################################################################################

# Get project configuration
config = configparser.ConfigParser()
config.read_file(open(sys.argv[1]))

# Open API to Azure blob store
azure_cred = configparser.ConfigParser()
azure_cred.read_file(open(os.environ['AZURE_SECRET']))
azure = sentinel2_azure.AzureAccess(azure_cred.get('account', 'user'),
                                    azure_cred.get('account', 'key'))

# Open classifier
bsc = Big_Sentinel_Classifier.SurfaceClassifier(os.environ['PROCESS_DIR'] + config.get('options', 'classifier'))

# set up empty lists and dataframes to append to
download_problem_list = []  # empty list to append details of skipped tiles due to missing info
QC_reject_list = []  # empty list to append details of skipped tiles due to cloud cover
good_tile_list = []  # empty list to append tiles used in analysis
masterDF = pd.DataFrame()
dates = []

# send config data to log file and report to console
print("WRITING PARAMS TO LOG FILE\n")
text_file = open("BISC_Param_Log.txt", "w")
text_file.write("PARAMS FOR BIG ICE SURF CLASSIFIER\n\n")
text_file.write("DATE/TIME (yyyy,mm,dd,hh,mm,ss) = {}\n".format(dt.datetime.now()))
text_file.write("DISORT CONFIG = {}\n".format(config.get('options','retrieve_disort_params')))
text_file.write("INTERPOLATION CONFIG = {}\n".format(config.get('options','interpolate_missing_tiles')))
text_file.write("ENERGY_BALANCE CONFIG = {}\n".format(config.get('options','calculate_melt')))
text_file.write("Thresholds: Cloud cover= {}, \nIce Area = {}\n".format(config.get('thresholds',
    'cloudCoverThresh'),config.get('thresholds','minArea')))
text_file.write("TILES: {}\n".format(config.get('options','tiles')))
text_file.write("YEARS: {}\n".format(config.get('options','years')))
text_file.write("MONTHS: {}\n".format(config.get('options','months')))
text_file.close()

print("RUNNING WITH THE FOLLOWING CONFIGURATION:")
print("DISORT CONFIG = ", config.get('options','retrieve_disort_params'))
print("FIGURE CONFIG = ", config.get('options','savefigs'))
print("INTERPOLATION CONFIG = ", config.get('options','interpolate_missing_tiles'))
print("ENERGY BALANCE CONFIG = ", config.get('options','interpolate_missing_tiles'))
print("TILES: ", config.get('options','tiles'), " YEARS: ", config.get('options','years'), "MONTHS: ", config.get('options','months'))
print("Thresholds: Cloud cover= ", config.get('thresholds','cloudCoverThresh'), " Ice Area = ", config.get('thresholds','minArea'))
print()

###################################################################################
######################### SET TILE AND DATE RANGE #################################
###################################################################################


years = json.loads(config.get('options', 'years'))
months = json.loads(config.get('options', 'months'))

# set up dates (this will create list of all days in year/month range specified above)

for year in years:
    for month in months:

        startDate = dt.date(year, month, 1)
        endDate = dt.date(year, month, calendar.monthrange(year, month)[1])
        dates_pd = pd.date_range(startDate, endDate, freq='1D')

        for date in dates_pd:
            dates.append(date.strftime('%Y%m%d'))


###################################################################################
############### RUN FUNCTIONS & HEALTHCHECKS ######################################
###################################################################################

tiles = json.loads(config.get('options', 'tiles'))

for tile in tiles:

    tile = tile.lower()  # azure blob store is case sensitive: force lower case
    
    # first create directory to save outputs to
    dirName = str(os.environ['PROCESS_DIR'] + '/outputs/' + tile + "/")

    # Create target Directory if it doesn't already exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")


    # make DirName the path to save files to
    savepath = dirName

    for date in dates:

        print("\n DOWNLOADING FILES: {} {}\n".format(tile,date))

        #query blob for files in tile and date range
        filtered_bloblist, download_flag = azure.download_imgs_by_date(tile,date, os.environ['PROCESS_DIR'])

        # check download and only proceed if correct no. of files and cloud layer present

        if download_flag:
            print("\nDOWNLOAD FLAG : Skipping {}, {}".format(tile,date))
            download_problem_list.append('{}_{}'.format(tile, date))


        else:
            print("\nChecking cloud, ice and NaN cover")

            Icemask, Cloudmask = sentinel2_tools.format_mask(os.environ['PROCESS_DIR'],
                                                             os.environ['PROCESS_DIR'] + config.get('options',
                                                                                                    'icemask'),
                                                             os.environ['PROCESS_DIR'] + '/outputs/ICE_MASK.nc',
                                                             int(config.get('thresholds', 'cloudCoverThresh')))

            QCflag, useable_area = sentinel2_tools.img_quality_control(os.environ['PROCESS_DIR'],
                                                                       Icemask, Cloudmask,
                                                                       int(config.get('thresholds', 'minArea')))

           
            
            # Check image is not too cloudy. If OK, proceed, if not, skip tile/date
            if QCflag:
                print("\n QC FlAG: Skipping {}, {}: {} % useable pixels".format(tile,date,np.round(useable_area,4)))
                QC_reject_list.append('{}_{}_useable_area = {}'.format(tile,date,np.round(useable_area,2)))



            else:
                print("\n NO FLAGS, proceeding with image analysis for {}, {}".format(tile,date))
                good_tile_list.append('{}_{}_useable_area = {} '.format(tile, date, np.round(useable_area,2)))

                s2xr = bsc.load_img_to_xr(os.environ['PROCESS_DIR'],
                                          int(config.get('options', 'resolution')),
                                          Icemask,
                                          Cloudmask)

                # apply classifier and calculate albedo
                predicted = bsc.classify_image(s2xr, savepath, tile, date, savefigs=True)
                albedo = bsc.calculate_albedo(s2xr)
                Index2DBA, predict2DBA = bsc.calculate_2DBA(s2xr)
                mask2 = bsc.combine_masks(s2xr)
                
                # 1) Retrieve projection info from S2 datafile and add to netcdf
                proj_info = xr_cf_conventions.create_grid_mapping(s2xr.Data.attrs['crs'])

                # 2) Create associated lat/lon coordinates DataArrays using georaster (imports geo metadata without loading img)
                # see georaster docs at https:/media.readthedocs.org/pdf/georaster/latest/georaster.pdf
                # find B02 jp2 file
                fileB2 = glob.glob(str(os.environ['PROCESS_DIR'] + '*{}*B02_20m.jp2').format(tile.upper()))
                fileB2 = fileB2[0]

                lon, lat = xr_cf_conventions.create_latlon_da(fileB2, 'x', 'y',
                                                              s2xr.x, s2xr.y, proj_info)

                # 3) add predicted map array and add metadata

                predicted = predicted.fillna(0)
                predicted = predicted.where(mask2 > 0)
                predicted.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
                predicted.name = 'Surface Class'
                predicted.attrs['long_name'] = 'Surface classified using Random Forest'
                predicted.attrs['units'] = 'None'
                predicted.attrs['key'] = config.get('netcdf', 'predicted_legend')
                predicted.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                # add albedo map array and add metadata

                albedo = albedo.fillna(0)
                albedo = albedo.where(mask2 > 0)
                albedo.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
                albedo.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
                albedo.attrs['units'] = 'dimensionless'
                albedo.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']


                # add 2DBA map array and add metadata

                Index2DBA = Index2DBA.fillna(0)
                Index2DBA = Index2DBA.where(mask2 > 0)
                Index2DBA.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
                Index2DBA.name = '2DBA band ratio index value (Wang et al. 2018)'
                Index2DBA.attrs['units'] = 'dimensionless'
                Index2DBA.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                # add 2DBA map array and add metadata

                predict2DBA = predict2DBA.fillna(0)
                predict2DBA = predict2DBA.where(mask2 > 0)
                predict2DBA.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
                predict2DBA.name = '2DBA band ratio index value (Wang et al. 2018)'
                predict2DBA.attrs['units'] = 'dimensionless'
                predict2DBA.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                ## RUN DISORT INVERSION
                #######################

                if config.get('options','retrieve_disort_params')=='True':

                    print("\n NOW RUNNING DISORT INVERSION FUNCTION \n")
                    
                    # run disort inversion

                    densities = [[400,400,400,400,400],[450,450,450,450,450],[500,500,500,500,500],\
                        [550,550,550,550,550],[600,600,600,600,600],[650,650,650,650,650],\
                            [700,700,700,700,700],[750,750,750,750,750],[800,800,800,800,800],\
                                [850,850,850,850,850],[900,900,900,900,900]]

                    side_lengths = [[500,500,500,500,500],[700,700,700,700,700],[900,900,900,900,900],[1100,1100,1100,1100,1100],
                    [1300,1300,1300,1300,1300],[1500,1500,1500,1500,1500],[2000,2000,2000,2000,2000],[3000,3000,3000,3000,3000],
                    [5000,5000,5000,5000,5000],[8000,8000,8000,8000,8000],[10000,10000,10000,10000,10000],
                    [15000,15000,15000,15000,15000]]

                    algae = [[0,0,0,0,0], [1000,0,0,0,0], [5000,0,0,0,0], [10000,0,0,0,0], [50000,0,0,0,0], [10000,0,0,0,0],\
                        [15000,0,0,0,0], [20000,0,0,0,0], [250000,0,0,0,0], [50000,0,0,0,0], [75000,0,0,0,0], [100000,0,0,0,0],\
                            [125000,0,0,0,0], [150000,0,0,0,0], [1750000,0,0,0,0], [200000,0,0,0,0], [250000,0,0,0,0]]

                    wavelengths = np.arange(0.3,5,0.01)

                    idx = [19, 26, 36, 40, 44, 48, 56, 131, 190]

                    bsc.invert_disort(s2xr,mask2,predicted,side_lengths,densities,algae,wavelengths,idx, tile, year, month)

                    # Add metadata to retrieved disort parameter arrays + mask
                    with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'side_lengths.nc')) as side_length:

                        side_length.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
                        side_length.name = "Grain size"
                        side_length.attrs['long_name'] = 'Grain size in microns. Assumed homogenous to 10 cm depth'
                        side_length.attrs['units'] = 'Microns'
                        side_length.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']
                    
                    with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'densities.nc')) as density:

                        density.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
                        density.name = "Density"
                        density.attrs['long_name'] = 'Ice column density in kg m-3. Assumed to be homogenous to 10 cm depth'
                        density.attrs['units'] = 'Kg m-3'
                        density.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                    with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'algae.nc')) as algae:

                        algae.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
                        algae.name = "Algae"
                        algae.attrs['long_name'] = 'Ice column algae in kg m-3. Assumed to be homogenous to 10 cm depth'
                        algae.attrs['units'] = 'Kg m-3'
                        algae.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                    # collate data arrays into a dataset
                    dataset = xr.Dataset({
                        'classified': (['x', 'y'], predicted),
                        'albedo': (['x', 'y'], albedo),
                        'Index2DBA':(['x','y'],Index2DBA),
                        'predict2DBA': (['x','y'],predict2DBA),
                        'grain_size': (['x','y'],side_length),
                        'density':(['x','y'],density),
                        'algae':(['x','y'],algae),
                        'Icemask': (['x', 'y'], s2xr.Icemask),
                        'Cloudmask': (['x', 'y'], s2xr.Cloudmask),
                        'FinalMask': (['x', 'y'], mask2),
                        proj_info.attrs['grid_mapping_name']: proj_info,
                        'longitude': (['x', 'y'], lon),
                        'latitude': (['x', 'y'], lat)
                    },
                        coords={'x': s2xr.x, 'y': s2xr.y})

                    dataset['algae'] = dataset['algae'].where(dataset.classified>=4)

                else:
                    # if disort retrieval is not selected in config/template file
                    # collate data arrays into a dataset
                    dataset = xr.Dataset({
                        'classified': (['x', 'y'], predicted),
                        'albedo': (['x', 'y'], albedo),
                        'Index2DBA':(['x','y'],Index2DBA),
                        'predict2DBA': (['x','y'],predict2DBA),
                        'Icemask': (['x', 'y'], s2xr.Icemask),
                        'Cloudmask': (['x', 'y'], s2xr.Cloudmask),
                        'FinalMask': (['x', 'y'], mask2),
                        proj_info.attrs['grid_mapping_name']: proj_info,
                        'longitude': (['x', 'y'], lon),
                        'latitude': (['x', 'y'], lat)
                    },
                        coords={'x': s2xr.x, 'y': s2xr.y})

                if config.get('options','interpolate_cloud')=='True':
                    dataset = sentinel2_tools.cloud_interpolator(dataset)

                # add geo info
                dataset = xr_cf_conventions.add_geo_info(dataset, 'x', 'y',
                                                         config.get('netcdf', 'author'),
                                                         config.get('netcdf', 'title'))

                # if toggles, interpolate over missing values due to cloud cover
                if config.get('options','interpolate_cloud')=='True':
                    dataset = sentinel2_tools.cloud_interpolator(dataset)

                dataset.to_netcdf(savepath + "{}_{}_Classification_and_Albedo_Data.nc".format(tile, date), mode='w', format='NETCDF4_CLASSIC')
                
                # flush dataset from disk
                dataset = None
                predicted = None
                albedo = None
                density = None
                algae = None

                # explicitly call garbage collector to deallocate memory
                print("GARBAGE COLLECTION\n")
                gc.collect()
                        
        # clear process directory 
        sentinel2_tools.clear_img_directory(os.environ['PROCESS_DIR'])

    # interpolate missing tiles if toggled ON
    if config.get('options','interpolate_missing_tiles')=='True':
        print("\nINTERPOLATING MISSING TILES")
        sentinel2_tools.imageinterpolator(years,months,tile,proj_info)

    dateList = sentinel2_tools.create_outData(tile,year,month,savepath)
    sentinel2_tools.createSummaryData(tile,year,month,savepath,dateList)

    # save logs to csv files
    print("\n SAVING QC LOGS TO TXT FILES")
    np.savetxt(str(savepath + "/aborted_downloads_{}_{}_{}.csv".format(tile,year,months)), download_problem_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath + "/rejected_by_qc_{}_{}_{}.csv".format(tile,year,months)), QC_reject_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath + "/good_tiles_{}_{}_{}.csv".format(tile,year,months)), good_tile_list, delimiter=",", fmt='%s')

print()
print("\nCOMPLETED RUN")