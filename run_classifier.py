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
import Ice_Surface_Analyser
import xr_cf_conventions
import multiprocessing as mp
import gc

###################################################################################
######## DEFINE BLOB ACCESS, GLOBAL VARIABLES AND SET UP EMPTY LISTS/ARRAYS #######
###################################################################################

download_problem_list = []  # empty list to append details of skipped tiles due to missing info
QC_reject_list = []  # empty list to append details of skipped tiles due to cloud cover
good_tile_list = []  # empty list to append tiles used in analysis
dates = []

# Get project configuration
config = configparser.ConfigParser()
config.read_file(open(sys.argv[1]))

# Open API to Azure blob store
azure_cred = configparser.ConfigParser()
azure_cred.read_file(open(os.environ['AZURE_SECRET']))
azure = sentinel2_azure.AzureAccess(azure_cred.get('account', 'user'),
                                    azure_cred.get('account', 'key'))

# Open classifier
isa = Ice_Surface_Analyser.SurfaceClassifier(os.environ['PROCESS_DIR'] + config.get('options', 'classifier'))

# send config data to log file and report to console
print("WRITING PARAMS TO LOG FILE\n")
text_file = open("BISC_Param_Log.txt", "w")
text_file.write("PARAMS FOR BIG ICE SURF CLASSIFIER\n\n")
text_file.write("DATE/TIME (yyyy,mm,dd,hh,mm,ss) = {}\n".format(dt.datetime.now()))
text_file.write("SNICAR CONFIG = {}\n".format(config.get('options','retrieve_snicar_params')))
text_file.write("INTERPOLATION CONFIG = {}\n".format(config.get('options','interpolate_missing_tiles')))
text_file.write("ENERGY_BALANCE CONFIG = {}\n".format(config.get('options','calculate_melt')))
text_file.write("Thresholds: Cloud cover= {}, \nIce Area = {}\n".format(config.get('thresholds',
    'cloudCoverThresh'),config.get('thresholds','minArea')))
text_file.write("TILES: {}\n".format(config.get('options','tiles')))
text_file.write("YEARS: {}\n".format(config.get('options','years')))
text_file.write("MONTHS: {}\n".format(config.get('options','months')))
text_file.close()

print("RUNNING WITH THE FOLLOWING CONFIGURATION:")
print("SNICAR CONFIG = ", config.get('options','retrieve_snicar_params'))
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
        filtered_bloblist, download_flag = azure.download_imgs_by_date(tile,
                                                                      date, os.environ['PROCESS_DIR'])

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


            #############################################
            ## IF HEALTHCHECKS OK, PROCEED WITH ANALYSIS
            #############################################
            else: # i.e. condition = healthcheck passed
                
                print("\n NO FLAGS, proceeding with image analysis for {}, {}".format(tile,date))
                good_tile_list.append('{}_{}_useable_area = {} '.format(tile, date, np.round(useable_area,2)))

                s2xr = isa.load_img_to_xr(os.environ['PROCESS_DIR'],
                                          int(config.get('options', 'resolution')),
                                          Icemask,
                                          Cloudmask)

                # apply classifier and calculate albedo
                predicted = isa.classify_image(s2xr, savepath, tile, date, savefigs=True)
                albedo = isa.calculate_albedo(s2xr)
                Index2DBA, predict2DBA = isa.calculate_2DBA(s2xr)
                mask2 = isa.combine_masks(s2xr)

                # 1) Retrieve projection info from S2 datafile and add to netcdf
                proj_info = xr_cf_conventions.create_grid_mapping(s2xr.Data.attrs['crs'])

                # 2) Create associated lat/lon coordinates DataArrays using georaster (imports geo metadata without loading img)
                # see georaster docs at https:/media.readthedocs.org/pdf/georaster/latest/georaster.pdf
                # find B02 jp2 file
                fileB2 = glob.glob(str(os.environ['PROCESS_DIR'] + "*{}*".format(tile.upper()) + '*B02_20m.jp2'))
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

                ## RUN SNICAR INVERSION
                #######################

                if config.get('options','retrieve_snicar_params')=='True':

                    print("\n NOW RUNNING SNICAR INVERSION FUNCTION \n")
                    
                    # run snicar inversion
                    # set wavelength range from start, stop, step defined in config
                    wlmin = config.get('options', 'wlmin'); wlmax = config.get('options', 'resolution'); 
                    wlstep = config.get('options', 'resolution')
                    wavelengths = np.arange(wlmin,wlmax,wlstep)
                    idx = config.get('options', 'idx') # get S2 indexes from config

                    # call snicar inversion from isa file
                    isa.invert_snicar_multi_LUT(s2xr, mask2, predicted, predict2DBA, wavelengths, idx, tile, year, month)

                    # Add metadata to retrieved snicar parameter arrays + mask
                    with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'density.nc')) as dens:

                        dens.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
                        dens.name = "density"
                        dens.attrs['long_name'] = 'density of surface'
                        dens.attrs['units'] = 'kg m-3'
                        dens.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']
                    

                    with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'dz.nc')) as dz:

                        dens.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
                        dens.name = "dz"
                        dens.attrs['long_name'] = 'thickness of unsaturated WC'
                        dens.attrs['units'] = 'meters'
                        dens.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                    with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'algae.nc')) as algae:

                        algae.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
                        algae.name = "Algae"
                        algae.attrs['long_name'] = 'Ice column algae in ppb.'
                        algae.attrs['units'] = 'ppb'
                        algae.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                    with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'reff.nc')) as grain:

                        grain.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
                        grain.name = "grain radius"
                        grain.attrs['long_name'] = 'grain effective radius at surface.'
                        grain.attrs['units'] = 'micron'
                        grain.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

                    # collate data arrays into a dataset
                    dataset = xr.Dataset({
                        'classified': (['x', 'y'], predicted),
                        'albedo': (['x', 'y'], albedo),
                        'density': (['x','y'], dens),
                        'dz':(['x','y'],dz),
                        'reff':(['x','y'], grain),
                        'algae':(['x','y'],algae),
                        'predict2DBA':(['x','y'],predict2DBA),

                        proj_info.attrs['grid_mapping_name']: proj_info,
                        'longitude': (['x', 'y'], lon),
                        'latitude': (['x', 'y'], lat)
                    },
                        coords={'x': s2xr.x, 'y': s2xr.y})

                    dataset['algae'] = dataset['algae'].where(dataset.classified>3)

                else:
                    # if snicar retrieval is not selected in config/template file
                    # collate data arrays into a dataset
                    dataset = xr.Dataset({
                        'classified': (['x', 'y'], predicted),
                        'albedo': (['x', 'y'], albedo),
                        'Icemask': (['x', 'y'], s2xr.Icemask),
                        'Cloudmask': (['x', 'y'], s2xr.Cloudmask),
                        'FinalMask': (['x', 'y'], mask2),
                        proj_info.attrs['grid_mapping_name']: proj_info,
                        'longitude': (['x', 'y'], lon),
                        'latitude': (['x', 'y'], lat)
                    },
                        coords={'x': s2xr.x, 'y': s2xr.y})

                # add geo info
                dataset = xr_cf_conventions.add_geo_info(dataset, 'x', 'y',
                                                         config.get('netcdf', 'author'),
                                                         config.get('netcdf', 'title'))


                # save netCDF to file
                dataset.to_netcdf(savepath + "{}_{}_Classification_and_Albedo_Data.nc".format(tile, date), mode='w', format='NETCDF4_CLASSIC')
                
                # flush disk and expicit garbage collection
                dataset = None
                predicted = None
                albedo = None
                algae = None
                grain = None
                dz = None
                dens = None
                gc.collect()
                        
        # clear process directory of raw S2 files 
        sentinel2_tools.clear_img_directory(os.environ['PROCESS_DIR'])

    # colate data into single netCDF and save to disk using function in sentinel2_tools.py
    dateList = sentinel2_tools.create_outData(tile,year,month,savepath)

    # save logs to csv files
    print("\n SAVING QC LOGS TO TXT FILES")
    np.savetxt(str(savepath + "/aborted_downloads_{}_{}_{}.csv".format(tile,year,months)), download_problem_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath + "/rejected_by_qc_{}_{}_{}.csv".format(tile,year,months)), QC_reject_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath + "/good_tiles_{}_{}_{}.csv".format(tile,year,months)), good_tile_list, delimiter=",", fmt='%s')


print()
print("\nCOMPLETED RUN")
