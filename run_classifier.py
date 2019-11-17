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
# TODO: devise some automated protection against interpolating cloudy pixels in sentinel2_tools.imageinterpolator()
# TODO: consider nearest-neighbour type spatial interpolation over cloudy pixels (could be part of previous todo)

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
text_file.write(f"DATE/TIME (yyyy,mm,dd,hh,mm,ss) {dt.datetime.now()}\n")
text_file.write(f"SNICAR CONFIG = {config.get('options','retrieve_snicar_params')}\n")
text_file.write(f"INTERPOLATION CONFIG = {config.get('options','interpolate_missing_tiles')}\n")
text_file.write(f"Thresholds: Cloud cover= {config.get('thresholds','cloudCoverThresh')}, \nIce Area = {config.get('thresholds','minArea')}\n")
text_file.write(f"TILES: {config.get('options','tiles')}\n")
text_file.write(f"YEARS: {config.get('options','years')}\n")
text_file.write(f"MONTHS: {config.get('options','months')}\n")
text_file.close()

print("RUNNING WITH THE FOLLOWING CONFIGURATION:")
print("SNICAR CONFIG = ", config.get('options','retrieve_snicar_params'))
print("FIGURE CONFIG = ", config.get('options','savefigs'))
print("INTERPOLATION CONFIG = ", config.get('options','interpolate_missing_tiles'))
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
        print(f"\n DOWNLOADING FILES: {tile} {date}\n")

        # query blob for files in tile and date range
        # filtered_bloblist, download_flag = azure.download_imgs_by_date(tile,
        #                                                                date, os.environ['PROCESS_DIR'])

        # # check download and only proceed if correct no. of files and cloud layer present

        # if download_flag:
        #     print(f"\nDOWNLOAD FLAG : Skipping {tile}, {date} ")
        #     download_problem_list.append('{}_{}'.format(tile, date))



        # else:
        #     print("\nChecking cloud, ice and NaN cover")

        #     Icemask, Cloudmask = sentinel2_tools.format_mask(os.environ['PROCESS_DIR'],
        #                                                      os.environ['PROCESS_DIR'] + config.get('options',
        #                                                                                             'icemask'),
        #                                                      os.environ['PROCESS_DIR'] + '/outputs/ICE_MASK.nc',
        #                                                      int(config.get('thresholds', 'cloudCoverThresh')))

        #     QCflag, useable_area = sentinel2_tools.img_quality_control(os.environ['PROCESS_DIR'],
        #                                                                Icemask, Cloudmask,
        #                                                                int(config.get('thresholds', 'minArea')))

           
            
        #     # Check image is not too cloudy. If OK, proceed, if not, skip tile/date
        #     if QCflag:
        #         print(f"\n QC FlAG: Skipping {tile}, {date}: {np.round(useable_area,4)} % useable pixels")
        #         QC_reject_list.append(f'{tile}_{date}_useable_area = {np.round(useable_area,2)}')



        #     else:
        #         print(f"\n NO FLAGS, proceeding with image analysis for {tile}, {date}")
        #         good_tile_list.append('{}_{}_useable_area = {} '.format(tile, date, np.round(useable_area,2)))

        #         s2xr = bsc.load_img_to_xr(os.environ['PROCESS_DIR'],
        #                                   int(config.get('options', 'resolution')),
        #                                   Icemask,
        #                                   Cloudmask)

        #         predicted = bsc.classify_image(s2xr, savepath, tile, date, savefigs=True)
        #         albedo = bsc.calculate_albedo(s2xr)
 
        #         ## Collate predicted map, albedo map and projection info into xarray dataset
        #         mask2 = bsc.combine_masks(s2xr)

        #         # 1) Retrieve projection info from S2 datafile and add to netcdf
        #         proj_info = xr_cf_conventions.create_grid_mapping(s2xr.Data.attrs['crs'])

        #         # 2) Create associated lat/lon coordinates DataArrays using georaster (imports geo metadata without loading img)
        #         # see georaster docs at https:/media.readthedocs.org/pdf/georaster/latest/georaster.pdf
        #         # find B02 jp2 file
        #         fileB2 = glob.glob(str(os.environ['PROCESS_DIR'] + '*B02_20m.jp2'))
        #         fileB2 = fileB2[0]

        #         lon, lat = xr_cf_conventions.create_latlon_da(fileB2, 'x', 'y',
        #                                                       s2xr.x, s2xr.y, proj_info)

        #         # 3) add predicted map array and add metadata
        #         predicted = predicted.fillna(0)
        #         predicted = predicted.where(mask2 > 0)
        #         predicted.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
        #         predicted.name = 'Surface Class'
        #         predicted.attrs['long_name'] = 'Surface classified using Random Forest'
        #         predicted.attrs['units'] = 'None'
        #         predicted.attrs['key'] = config.get('netcdf', 'predicted_legend')
        #         predicted.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

        #         # add albedo map array and add metadata
        #         albedo = albedo.fillna(0)
        #         albedo = albedo.where(mask2 > 0)
        #         albedo.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
        #         albedo.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
        #         albedo.attrs['units'] = 'dimensionless'
        #         albedo.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

        #         ## RUN SNICAR INVERSION
        #         #######################

        #         if config.get('options','retrieve_snicar_params')=='True':

        #             print("\n NOW RUNNING SNICAR INVERSION FUNCTION \n")
                    
        #             # run snicar inversion
        #             bsc.invert_snicar(s2xr,mask2, predicted)

        #             # Add metadata to retrieved snicar parameter arrays + mask
        #             with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'side_lengths.nc')) as side_length:
        #                 side_length.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
        #                 side_length.name = "Grain size"
        #                 side_length.attrs['long_name'] = 'Grain size in microns. Assumed homogenous to 10 cm depth'
        #                 side_length.attrs['units'] = 'Microns'
        #                 side_length.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']
                    
        #             with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'densities.nc')) as density:
        #                 density.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
        #                 density.name = "Density"
        #                 density.attrs['long_name'] = 'Ice column density in kg m-3. Assumed to be homogenous to 10 cm depth'
        #                 density.attrs['units'] = 'Kg m-3'
        #                 density.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

        #             with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'dust.nc')) as dust:
        #                 dust.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
        #                 dust.name = "Dust"
        #                 dust.attrs['long_name'] = 'Dust mass mixing ratio in upper 1mm of ice column'
        #                 dust.attrs['units'] = 'ng/g or ppb'
        #                 dust.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']

        #             with xr.open_dataarray(str(os.environ['PROCESS_DIR'] + 'algae.nc')) as algae:
        #                 algae.encoding = {'dtype': 'float16', 'zlib': True, '_FillValue': -9999}
        #                 algae.name = "Algae"
        #                 algae.attrs['long_name'] = 'Ice column algae in kg m-3. Assumed to be homogenous to 10 cm depth'
        #                 algae.attrs['units'] = 'Kg m-3'
        #                 algae.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']


        #             # collate data arrays into a dataset
        #             dataset = xr.Dataset({
        #                 'classified': (['x', 'y'], predicted),
        #                 'albedo': (['x', 'y'], albedo),
        #                 'grain_size': (['x','y'],side_length),
        #                 'density':(['x','y'],density),
        #                 'dust':(['x','y'],dust),
        #                 'algae':(['x','y'],algae),
        #                 'Icemask': (['x', 'y'], s2xr.Icemask),
        #                 'Cloudmask': (['x', 'y'], s2xr.Cloudmask),
        #                 'FinalMask': (['x', 'y'], mask2),
        #                 proj_info.attrs['grid_mapping_name']: proj_info,
        #                 'longitude': (['x', 'y'], lon),
        #                 'latitude': (['x', 'y'], lat)
        #             },
        #                 coords={'x': s2xr.x, 'y': s2xr.y})

        #         else:
        #             # if snicar retrieval is not selected in config/template file
        #             # collate data arrays into a dataset
        #             dataset = xr.Dataset({
        #                 'classified': (['x', 'y'], predicted),
        #                 'albedo': (['x', 'y'], albedo),
        #                 'Icemask': (['x', 'y'], s2xr.Icemask),
        #                 'Cloudmask': (['x', 'y'], s2xr.Cloudmask),
        #                 'FinalMask': (['x', 'y'], mask2),
        #                 proj_info.attrs['grid_mapping_name']: proj_info,
        #                 'longitude': (['x', 'y'], lon),
        #                 'latitude': (['x', 'y'], lat)
        #             },
        #                 coords={'x': s2xr.x, 'y': s2xr.y})

        #         if config.get('options','calculate_melt')=='True':

        #             bsc.energy_balance(tile, date, savepath, mask2, S2vals)
        #             dataset['melt'] = xr.open_dataarray(str(os.environ['PROCESS_DIR'] + {tile} + f"MELT_{tile}_{date}.nc"))



        #         # add geo info
        #         dataset = xr_cf_conventions.add_geo_info(dataset, 'x', 'y',
        #                                                  config.get('netcdf', 'author'),
        #                                                  config.get('netcdf', 'title'))

        #         dataset.to_netcdf(savepath + "{}_{}_Classification_and_Albedo_Data.nc".format(tile, date), mode='w')

        #         # flush dataset from disk
        #         dataset = None
        #         predicted = None
        #         albedo = None
        #         density = None
        #         dust = None
        #         algae = None

        #         # generate summary data
        #         summaryDF = bsc.albedo_report(tile, date, savepath)
        
        # # clear process directory 
        # sentinel2_tools.clear_img_directory(os.environ['PROCESS_DIR'])

    # interpolate missing tiles if toggled ON
    if config.get('options','interpolate_missing_tiles')=='True':
        print("\nINTERPOLATING MISSING TILES")
        sentinel2_tools.imageinterpolator(years,months,tile)

    # collate individual dates into single dataset for each tile 
    print("\nCOLLATING INDIVIDUAL TILES INTO FINAL DATASET")
    concat_dataset = bsc.concat_all_dates(savepath, tile)

    # save logs to csv files
    print("\n SAVING QC LOGS TO TXT FILES")
    np.savetxt(str(savepath + "/aborted_downloads.csv"), download_problem_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath + "/rejected_by_qc.csv"), QC_reject_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath + "/good_tiles.csv"), good_tile_list, delimiter=",", fmt='%s')

print()
print("\nCOMPLETED RUN")
