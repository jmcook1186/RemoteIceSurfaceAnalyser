"""
Functions for local-system management of Sentinel-2 files.

"""

from collections import OrderedDict
import os
import sys
import shutil
from osgeo import gdal
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib as mpl
import configparser
import pandas as pd
from scipy import interpolate
import sentinel2_azure
plt.ioff()

# Get project configuration
config = configparser.ConfigParser()
config.read_file(open(sys.argv[1]))


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
    metadata = api.to_dataframe(products)
    L1Cfiles = metadata.filename
    print('\n',' DOWNLOADING {} FILES '.center(80, 'X').format(len(L1Cfiles)),'\n', 'X'.center(80, 'X'))

    # download all files
    api.download_all(products, directory_path = L1Cpath)

    return L1Cfiles


def process_L1C_to_L2A(L1C_path, L1Cfiles, S2_resolution, unzip_files=True):
    """
    This function is takes the downloaded L1C products from SentinelHub and converts them to L2A product using Sen2Cor.
    This is achieved as a batch job by iterating through the file names (L1Cfiles) stored in L1Cpath
    Running this script will save the L2A products to the working directory

    Requires 'SEN2COR_BIN' environment variable to be set

    :return: None

    """

    sen2cor_bin = os.environ['SEN2COR_BIN']
    if len(sen2cor_bin) == 0:
        print('Error: sen2cor bin path environment variable is not set.')
        raise OSError

    for L1C in L1Cfiles:

        # unzip downloaded files in place, then delete zipped folders
        if unzip_files:

            unzip = str('unzip '+ L1C_path + L1C[0:-4]+'zip'+' -d ' + L1C_path)
            os.system(unzip)

        # process L1C to L2A
        try:
            sen2cor = str(sen2cor_bin + '/L2A_Process ' + L1C_path + '{}'.format(L1C) + ' --resolution={}'.format(S2_resolution))
            os.system(sen2cor)

        except:
            print("Processing {} L1C to L2A failed. Skipping to next tile".format(L1C))
    return



def remove_L1C(L1Cpath):
    """
    function removes L1C files from wdir after processing to L2A

    :param L1Cpath: path to wdir
    :return:

    """

    files = glob.glob(str(L1Cpath+'*L1C*')) # find files containing "L1C"

    for f in files:
        try:
            shutil.rmtree(f) # try removing using shutil.rmtree() (for unzipped files)
        except:
            os.remove(f) # if it can't be removed by shutil then use os.remove()

    return


def remove_L2A(L1Cpath, upload_status): # delete unused L1C products

    """
    Once files have been sent to blob, remove them from local storage

    :param L1Cpath: path to wdir
    :return:
    """

    files = glob.glob(str(L1Cpath+'*L2A*')) # find files containing "L1C"

    if upload_status == 1:
        for f in files:
            try:
                shutil.rmtree(f) # try removing using shutil.rmtree() (for unzipped files)
            except:
                os.remove(f) # if it can't be removed by shutil it is likely zipped - use os.remove()

    else:
        print('\n', ' S2A FILES NOT UPLOADED TO BLOB: S2A FILES NOT DELETED FROM LOCAL STORAGE '.center(80, 'X'), '\n',
              'X'.center(80, 'X'))

    return



def clear_img_directory(img_path):

    """
    Function deletes all files in the local image directory ready to download the next set. Outputs are all stored in
    separate output folder.
    :param img_path: path to working directory where img files etc are stored.
    :return: None
    """
    files = glob.glob(str(img_path+'*.jp2'))

    for f in files:
        os.remove(f)

    return



def format_mask(img_path, Icemask_in, Icemask_out, cloudProbThreshold):

    """
    Function to format the land/ice and cloud masks.
    First, the Greenland Ice Mapping Project (GIMP) mask is reprojected to match the coordinate system of the S2 files.
    The relevant tiles of the GIMP mask were stitched to create one continuous Boolean array in a separate script.
    The cloud mask is derived from the clpoud layer in the L2A product which is an array of probabilities (0 - 1) that
    each pixel is obscured by cloud. The variable cloudProbThreshold is a user defined value above which the pixel is
    given value 1, and below which it is given value 0, creating a Boolean cloud/not-cloud mask.
    Note that from 2016 onwards, the file naming convention changed in the Sentinel archive, with the string "CLD_20m"
    replaced by "CLDPRB_20m". Therefore an additional wildcard * was added to the search term *CLD*_20m.jp2.

    :param img_path: path to image to use as projection template
    :param Icemask_in: file path to mask file
    :param Icemask_out: file path to save reprojected mask
    :param cloudProbThreshold: threshold probability of cloud for masking pixel
    :return Icemask: Boolean to mask out pixels outside the ice sheet boundaries
    :return Cloudmask: Boolean to mask out pixels obscured by cloud
    """
    cloudmaskpath_temp = glob.glob(str(img_path + '*CLD*_20m.jp2')) # find cloud mask layer in filtered S2 image directory
    cloudmaskpath = cloudmaskpath_temp[0]

    mask = gdal.Open(Icemask_in)
    mask_proj = mask.GetProjection()
    mask_geotrans = mask.GetGeoTransform()
    data_type = mask.GetRasterBand(1).DataType
    n_bands = mask.RasterCount

    S2filename = glob.glob(str(img_path + '*B02_20m.jp2')) # use glob to find files because this allows regex such as * - necessary for iterating through downloads
    Sentinel = gdal.Open(S2filename[0]) # open the glob'd filed in gdal

    Sentinel_proj = Sentinel.GetProjection()
    Sentinel_geotrans = Sentinel.GetGeoTransform()
    w = Sentinel.RasterXSize
    h = Sentinel.RasterYSize

    mask_filename = Icemask_out
    new_mask = gdal.GetDriverByName('GTiff').Create(mask_filename,
                                                     w, h, n_bands, data_type)
    new_mask.SetGeoTransform(Sentinel_geotrans)
    new_mask.SetProjection(Sentinel_proj)

    gdal.ReprojectImage(mask, new_mask, mask_proj,
                        Sentinel_proj, gdal.GRA_NearestNeighbour)

    new_mask = None  # Flush disk

    maskxr = xr.open_rasterio(Icemask_out)
    mask_squeezed = xr.DataArray.squeeze(maskxr,'band')
    Icemask = xr.DataArray(mask_squeezed.values)

    # set up second mask for clouds
    Cloudmask = xr.open_rasterio(cloudmaskpath)
    Cloudmask = xr.DataArray.squeeze(Cloudmask,'band')
    # set pixels where probability of cloud < threshold to 0
    Cloudmask = Cloudmask.where(Cloudmask.values >= cloudProbThreshold, 0)
    Cloudmask = Cloudmask.where(Cloudmask.values < cloudProbThreshold, 1)

    return Icemask, Cloudmask



def img_quality_control(img_path, Icemask, Cloudmask, minimum_useable_area):

    """
    Function assesses image quality and raises flags if the image contains too little ice (i.e. mostly ocean, dry land
    or NaNs) or too much cloud cover.

    :param Icemask: Boolean for masking non-ice areas
    :param Cloudmask: Boolean for masking out cloudy pixels
    :param CloudCoverThreshold: threshold value for % of total pixels obscured by cloud. If over threshold image not used
    :param IceCoverThreshold: threshold value for % of total pixels outside ice sheet boundaries. If over threshold image not used
    :param NaNthreshold: threshold value for % of total pixels comprised by NaNs. If over threshold image not used
    :return QCflag: quality control flag. Raised if cloud, non-ice or NaN pixels exceed threshold values.
    :return CloudCover: % of total image covered by cloud
    :return IceCover: % of total image covered by ice
    :return NaNcover: % of total image comprising NaNs
    """
    
    print("CHECKING DATA QUALITY")

    S2file = glob.glob(str(img_path + '*B02_20m.jp2')) # find band2 image (fairly arbitrary choice of band image)

    with xr.open_rasterio(S2file[0]) as S2array: #open file

        total_pixels = S2array.size
        S2array = S2array.values # values to array
        S2array[S2array > 0] = 1 # make binary (i.e all positive values become 1s)
        S2array = np.squeeze(S2array) # reshape to match ice and cloudmasks (2d)
        count_nans = total_pixels - np.count_nonzero(S2array)
        percent_nans = (count_nans/total_pixels)*100
        percent_non_nans = 100 - percent_nans

        Icemask_inv = 1-Icemask
        S2array_inv = 1-S2array

        # create xarray dataset with ice mask, cloud mask and NaN mask.
        qc_ds = xr.Dataset({'Cloudmask': (('y','x'), Cloudmask.values),'Icemask': (('y','x'), Icemask_inv),
                            'NaNmask': (('y','x'), np.flip(S2array_inv,0))})

        # good pixels are zeros in all three layers, so if the sum of the three layers is >0, that pixel is bad because of
        # either NaN, non-ice or cloud
        ds_sum = qc_ds.Cloudmask.values+qc_ds.Icemask.values+qc_ds.NaNmask.values
        bad_pixels = np.count_nonzero(ds_sum)
        good_pixels = total_pixels - bad_pixels
        unuseable_area = (bad_pixels/total_pixels)*100
        useable_area = (good_pixels/total_pixels)*100
        print('good = {}%, bad = {}%'.format(useable_area,unuseable_area))

    # report to console
    print(f"{np.round(useable_area,2)} % of the image is composed of useable pixels")
    print("FINISHED QC CHECKS")

    # raise flags
    if (useable_area < minimum_useable_area):

        QCflag = True
        print("USEABLE PIXELS < MINIMUM THRESHOLD: DISCARDING IMAGE \n")

    else:

        QCflag = False
        print("SUFFICIENT GOOD PIXELS: PROCEEDING WITH IMAGE ANALYSIS \n")

    return QCflag, useable_area
    

def imageinterpolator(years, months, tile, proj_info):
    """
    Function identifies the missing dates in the image repository - i.e those days where S2 images are not availabe due to
    cloud cover, insufficient ice coverage, a download problem or simply lack of S2 overpass. For each missing date, this
    function finds the closest previous and closest future images stored in the repository and applies a linear inteprolation
    between the past and future values for each pixel. This linear regression is then used to predict the pixel values for the
    missing date. In this way, a synthetic image is created and added to the image repository to infill the missing dates
    and provide a "complete" record.

    param: years (as defined in template)
    param: months (as defined in template)
    param: tile (as defined in template)

    returns: None, but saves new image (png) and dataset (.nc) to the output path

    """

    print("\nSTARTING INTERPOLATION FUNCTION\n")

    # define first and last date of entire season (i.e. 1st and last across all dates in model run)
    seasonStart = str(str(years[0]) + '_' + str(months[0]) + '_01')

    # since it is JJA, only June has 30 days, July and August have 31
    if months[-1] == 6:
        seasonEnd = str(str(years[-1]) + '_' + str(months[-1]) + '_30')
    else:
        seasonEnd = str(str(years[-1]) + '_' + str(months[-1]) + '_31')

    # get list of "good" nc files in process directory and extract the dates as strings
    DateList = glob.glob(str(os.environ['PROCESS_DIR'] + '/outputs/' + tile + '/' + tile + '_*.nc'))
    fmt = '%Y_%m_%d'  # set format for date string
    DOYlist = []  # empty list ready to receive days of year

    # create list of all DOYs between start and end of season
    DOYstart = dt.datetime.strptime(seasonStart, fmt).timetuple().tm_yday  # convert seasonStart to numeric DOY
    DOYend = dt.datetime.strptime(seasonEnd, fmt).timetuple().tm_yday  # convert seasonEnd to numeric DOY
    FullSeason = np.arange(DOYstart, DOYend, 1) # create list starting at DOY for seasonStart and ending at DOY for seasonEnd

    # Now create list of DOYs and list of strings for all the "good" dates in the image repo
    # strip out the date from the filename and insert underscores to separate out YYYY, MM, DD so formats are consistent
    # Note that the paths are different on the local and virtual machines, so the date srae in different positions
    for i in np.arange(0, len(DateList), 1):

        if config.get('options','vm')=='True':
            DateList[i] = DateList[i][65:-34]
            DateList[i] = str(DateList[i][0:4] + '_' + DateList[i][4:6] + '_' + DateList[i][
                                                                            6:8])  # format string to match format defined above
            DOYlist.append(dt.datetime.strptime(DateList[i], fmt).timetuple().tm_yday) # list of DOYs

        else:
            DateList[i] = DateList[i][69:-33]
            DateList[i] = str(DateList[i][0:4] + '_' + DateList[i][4:6] + '_' + DateList[i][
                                                                            6:8])  # format string to match format defined above
            DOYlist.append(dt.datetime.strptime(DateList[i], fmt).timetuple().tm_yday) # list of DOYs

    # compare full season DOY list with DOY list in image repo to identify only the missing dates between seasonStart and seasonEnd
    DOYlist = np.array(DOYlist)
    MissingDates = np.setdiff1d(FullSeason,DOYlist)  # setdiff1d identifies the dates present in Fullseason but not DOYlist
    year = seasonStart[0:4]  # the year is the first 4 characters in the string and is needed later

    print("\nMissing DOYs: ", MissingDates)

    # set up loop to create a new "synthetic" image for each missing date

    for Missing in MissingDates:

        print(f"\nGenerating data for DOY {Missing}")
        # for each missing date find the nearest past and future "good" images in the repo
        temp = DOYlist - Missing  # subtract the missing DOY from each DOY in the image repo
        closestPast = DOYlist[np.where(temp < 0, temp, -9999).argmax()]  # find the closest image from the past
        closestFuture = DOYlist[np.where(temp > 0, temp, 9999).argmin()]  # find the closest image in the future


        closestPastString = dt.datetime.strptime(str(year[2:4] + str(closestPast)),
                                                 '%y%j').date().strftime('%Y%m%d')  # convert to string format
        closestFutureString = dt.datetime.strptime(str(year[2:4] + str(closestFuture)),
                                                   '%y%j').date().strftime('%Y%m%d')  # convert to string format
        MissingDateString = dt.datetime.strptime(str(year[2:4] + str(Missing)),
                                                 '%y%j').date().strftime('%Y%m%d')  # convert to string format
        
        # report past, missing and future dates to console
        print(f"closestPast = {closestPastString}, closestFuture = {closestFutureString}")
        
        if (closestPastString > seasonStart) and (closestFutureString < seasonEnd): # greater than == nearer to present,
            # ensures interpolation does not try to go outside of available dates
            print(f"skipping date ({closestPastString} or {closestFutureString} out of range)")

        else:

            # load past and future images that will be used for interpolation
            imagePast = xr.open_dataset(str(
                os.environ['PROCESS_DIR'] + '/outputs/' + tile + '/' + tile + '_' + closestPastString + '_Classification_and_Albedo_Data.nc'))
            imageFuture = xr.open_dataset(str(
                os.environ['PROCESS_DIR']+'/outputs/' + tile + '/' + tile + '_' + closestFutureString + '_Classification_and_Albedo_Data.nc'))

            # conditionally include or exclude grain size, density, dust and algae 
            if config.get('options','retrieve_snicar_params')=='True':

                # extract the past and future albedo and classified layers.
                albPast = imagePast.albedo.values
                albFuture = imageFuture.albedo.values
                classPast = imagePast.classified.values
                classFuture = imageFuture.classified.values
                grainPast = imagePast.grain_size.values
                grainFuture = imageFuture.grain_size.values
                densityPast = imagePast.density.values
                densityFuture = imageFuture.density.values
                dustPast = imagePast.dust.values
                dustFuture = imageFuture.dust.values
                algaePast = imagePast.algae.values
                algaeFuture = imageFuture.algae.values

                # create mask
                maskPast = np.isnan(albPast)
                maskFuture = np.isnan(albFuture)
                combinedMask = np.ma.mask_or(maskPast, maskFuture)

                filenames = ['albedo','class','grain', 'density', 'dust', 'algae']
                counter = 0
                
                # loop through params calculating linear regression
                for i,j in [(albPast,albFuture),(classPast,classFuture),(grainPast,grainFuture),(densityPast,densityFuture),(dustPast,dustFuture),(algaePast,algaeFuture)]:                 

                    if counter == 1: # different function required for classification as it is not a continuous scale
                        
                        # albedo change is used to determine whether the past or future class is assigned
                        # so first calculate change in albedo
                        albedoDiffs = albPast - albFuture
                        albedoDiffs = albedoDiffs*0.5
                        albedoDiffsPredicted = albAPast - newImage

                        newClassImage = np.where(albedoDiffsPredicted > albedoDiffs, 
                        classFuture, classPast)

                        newClassImage = np.ma.array(newClassImage, mask=combinedMask)
                        newClassImage = newClassImage.filled(np.nan)


                        newClassImagexr = xr.DataArray(newClassImage)
                        newClassImage = None
                        newClassImagexr.to_netcdf(str(os.environ['PROCESS_DIR']+'interpolated_{}.nc'.format(filenames[counter])), mode='w')
                        newClassImagexr = None

                    else:
                        slopes = (i - j) / (closestFuture - closestPast)
                        intercepts = i - (slopes * closestPast)
                        newImage = slopes * Missing + intercepts
                        slopes = None
                        intercepts = None

                        # apply mask to synthetic images
                        newImage = np.ma.array(newImage, mask=combinedMask)
                        newImage = newImage.filled(np.nan)
                        newImage = abs(newImage)

                        if counter ==0: # for albedo layer, ensure values in range
                            newImage[newImage <= 0] = 0.00001 # ensure no interpolated values can be <= 0 
                            newImage[newImage >= 1] = 0.99999 # ensure no inteprolated values can be >=1
                        
                        newImagexr = xr.DataArray(newImage)
                        newImagexr.to_netcdf(str(os.environ['PROCESS_DIR']+'interpolated_{}.nc'.format(filenames[counter])), mode='w')
                        newImagexr = None
                        counter +=1                   

                with xr.open_dataarray(str(os.environ['PROCESS_DIR']+'interpolated_albedo.nc')) as albedo:
                    with xr.open_dataarray(str(os.environ['PROCESS_DIR']+'interpolated_class.nc')) as surfclass:
                        with xr.open_dataarray(str(os.environ['PROCESS_DIR']+'interpolated_grain.nc')) as grain:
                            with xr.open_dataarray(str(os.environ['PROCESS_DIR']+'interpolated_density.nc')) as density:
                                with xr.open_dataarray(str(os.environ['PROCESS_DIR']+'interpolated_dust.nc')) as dust:
                                    with xr.open_dataarray(str(os.environ['PROCESS_DIR']+'interpolated_algae.nc')) as algae:

                                        # collate data into xarray dataset and copy metadata from PAST
                                        newXR = xr.Dataset({
                                            'classified': (['x', 'y'], surfclass.values),
                                            'albedo': (['x', 'y'], albedo.values),
                                            'grain_size': (['x', 'y'], grain.values),
                                            'density': (['x', 'y'], density.values),
                                            'dust': (['x', 'y'], dust.values),
                                            'algae': (['x', 'y'], algae.values),
                                            'Icemask': (['x', 'y'], imagePast.Icemask),
                                            'Cloudmask': (['x', 'y'], imagePast.Cloudmask),
                                            'FinalMask': (['x', 'y'], combinedMask),
                                            proj_info.attrs['grid_mapping_name']: proj_info,
                                            'longitude': (['x', 'y'], imagePast.longitude),
                                            'latitude': (['x', 'y'], imagePast.latitude)
                                        }, coords = {'x': imagePast.x, 'y': imagePast.y})


            else: # if snicar retrieval toggled off

                # extract the past and future albedo and classified layers.
                albPast = imagePast.albedo.values
                albFuture = imageFuture.albedo.values


                # create mask
                maskPast = np.isnan(albPast)
                maskFuture = np.isnan(albFuture)
                combinedMask = np.ma.mask_or(maskPast, maskFuture)

                filenames = ['albedo','class']
                counter = 0

                classPast = imagePast.classified.values
                classFuture = imageFuture.classified.values

                # loop through params calculating linear regression
                for i,j in [(albPast,albFuture),(classPast,classFuture)]:

                    if counter == 1:
                        # albedo change is used to determine whether the past or future class is assigned
                        # so first calculate change in albedo
                        albedoDiffs = albPast - albFuture
                        albedoDiffs = albedoDiffs*0.5
                        albedoDiffsPredicted = albPast - newImage
                        newClassImage = np.where(albedoDiffsPredicted > albedoDiffs, classFuture, classPast)
                        classPast = None
                        classFuture = None
                        newClassImage = np.ma.array(newClassImage, mask=combinedMask)
                        newClassImage = newClassImage.filled(np.nan)
                        newClassImagexr = xr.DataArray(newClassImage)
                        newClassImage = None
                        newClassImagexr.to_netcdf(str(os.environ['PROCESS_DIR']+'interpolated_{}.nc'.format(filenames[counter])))
                        newClassImagexr = None

                    
                    else:
                        slopes = (i - j) / (closestFuture - closestPast)
                        intercepts = i - (slopes * closestPast)
                        newImage = slopes * Missing + intercepts

                        # apply mask to synthetic images
                        newImage = np.ma.array(newImage, mask=combinedMask)
                        newImage = newImage.filled(np.nan)
                        newImage = abs(newImage)

                        if counter ==0: # for albedo layer, ensure values in range
                            newImage[newImage <= 0] = 0.00001 # ensure no interpolated values can be <= 0 
                            newImage[newImage >= 1] = 0.99999 # ensure no inteprolated values can be >=1
                        
                        newImagexr = xr.DataArray(newImage)
                        newImagexr.to_netcdf(str(os.environ['PROCESS_DIR']+f'interpolated_{filenames[counter]}.nc'))
                        newImagexr = None
                        counter +=1


                with xr.open_dataarray(str(os.environ['PROCESS_DIR']+'interpolated_albedo.nc')) as albedo:
                    with xr.open_dataarray(str(os.environ['PROCESS_DIR']+'interpolated_class.nc')) as surfClass:


                        # collate data into xarray dataset and copy metadata from PAST
                        newXR = xr.Dataset({
                            'classified': (['x', 'y'], surfclass.values),
                            'albedo': (['x', 'y'], albedo.values),
                            'Icemask': (['x', 'y'], imagePast.Icemask),
                            'Cloudmask': (['x', 'y'], imagePast.Cloudmask),
                            'FinalMask': (['x', 'y'], combinedMask),
                            proj_info.attrs['grid_mapping_name']: proj_info,
                            'longitude': (['x', 'y'], imagePast.longitude),
                            'latitude': (['x', 'y'], imagePast.latitude)
                        }, coords = {'x': imagePast.x, 'y': imagePast.y})


                files = glob.glob(str(os.environ['PROCESS_DIR'] + 'interpolated_' + '*.nc'))
                for f in files:
                    os.remove(f)

            # save synthetic image in same format in sub directory as original "good" images
            newXR.to_netcdf(str(os.environ["PROCESS_DIR"] + "/outputs/" + tile + "/{}_{}_Classification_and_Albedo_Data.nc".format(tile, MissingDateString)), mode='w')            
            print("Interpolated dataset saved locally")
            
            newXR = None

    return


def create_outData(tile,year,month,savepath):

    outPath = str(savepath)
    dateList = []

    # GRAB FILES TO PROCESS
    file_list = glob.glob(outPath+'*Albedo_Data.nc')
    file_list = sorted(file_list)

    # if toggled, downsample the outData to the required resolution
    if config.get('options','downsample_outdata')=='True':
        outDataRes = config.get('options','outData_resolution')
        file_list = file_list[0:len(file_list):outDataRes]
    
    # GRAB DATES FROM FILE STRINGS
    for i in file_list:
        dateList.append(i.split(str(tile+'_'))[1].split('_Class')[0])

    # OPEN ALL FLES INTO ONE DATASET, SAVE CONCATENATED FILE TO NETCDF
    ds = xr.open_mfdataset(file_list,concat_dim='date',chunks={'x': 2000, 'y': 2000})
    ds = ds.assign_coords(date=dateList)

    ds.to_netcdf(str(outPath+'/FULL_OUTPUT_{}_{}.nc'.format(tile,year)))
    ds = None

    ### HERE DELETE INDIVIDUAL FILES
    for f in file_list:
        os.remove(f)

    return dateList


def createSummaryData(tile,year,month, savepath, dateList):

    outPath = savepath

    ds = xr.open_dataset(str(outPath+'/FULL_OUTPUT_{}_{}.nc'.format(tile,year)))
    
    # DEFINE SIZE OF OUT ARRAYS
    if config.get('options','retrieve_snicar_params')=='True':
        out = np.zeros(shape=(10,len(ds.date)))
        outClass = np.zeros(shape=(11,len(ds.date),7))

    else:
        out = np.zeros(shape=(2,len(ds.date)))
        outClass = np.zeros(shape=(3,len(ds.date),7))


    # START SUMMARIZING DATA AND APPENDING RESULTS TO OUT ARRAYS 
    if config.get('options','retrieve_snicar_params')=='True': 
        
        for i in range(len(ds.date)):

            date_i = dateList[i]
            # scalar outputs
            out[0,i] = ds.albedo.sel(date=date_i).mean().values  # mean albedo across whole tile
            out[1,i] = ds.albedo.sel(date=date_i).std().values # std albedo across whole tile
            out[2,i] = ds.grain_size.sel(date=date_i).mean().values # mean grain size
            out[3,i] = ds.grain_size.sel(date=date_i).std().values # std grain size
            out[4,i] = ds.density.sel(date=date_i).mean().values # mean density
            out[5,i] = ds.density.sel(date=date_i).std().values # std density
            out[6,i] = ds.algae.sel(date=date_i).mean().values # mean algae
            out[7,i] = ds.algae.sel(date=date_i).std().values # mean algae
            out[8,i] = ds.dust.sel(date=date_i).mean().values #mean dust
            out[9,i] = ds.dust.sel(date=date_i).std().values #std dust
            
            # outputs by class
            for j in range(7):

                outClass[0,i,j] = len(ds.classified.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # count instances of each class
                outClass[1,i,j] = np.mean(ds.classified.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # mean of albedo in each class
                outClass[2,i,j] = np.std(ds.classified.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # std of albedo in each class
                outClass[3,i,j] = np.mean(ds.grain_size.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # mean of grain size in each class
                outClass[4,i,j] = np.std(ds.grain_size.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # std of grain size in each class
                outClass[5,i,j] = np.mean(ds.density.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # mean of density in each class
                outClass[6,i,j] = np.std(ds.density.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # std of density in each class
                outClass[7,i,j] = np.mean(ds.algae.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # mean of algae in each class
                outClass[8,i,j] = np.std(ds.algae.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # std of algae in each class
                outClass[9,i,j] = np.mean(ds.dust.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # mean of dust in each class
                outClass[10,i,j] = np.std(ds.dust.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # std of dust in each class
        
        outTileXR = xr.DataArray(out,dims=('var','date'),coords={'var':['meanAlbedo','STDAlbedo','meanGrain','STDGrain','meanDensity','STDDensity','meanAlgae','STDAlgae','meanDust','STDDust'],'date':dateList})
        outClassXR = xr.DataArray(outClass,dims=('var','date','classID'),coords={'var': ['ClassCount','AlbedoMean','AlbedoSTD','GrainMean','GrainSTD','DensityMean','DensitySTD','AlgaeMean','AlgaeSTD','DustMean','DustSTD'], 'date':dateList,\
            'classID':['NONE','SN', 'WAT', 'CC', 'CI', 'LA', 'HA']})

        outTileXR.to_netcdf(savepath + 'OutData_{}_{}.nc'.format(tile,year))
        outClassXR.to_netcdf(savepath + 'OutData_{}_{}_byClass.nc'.format(tile,year))


    else:
        
        for i in range(len(ds.date)):
            
            date_i = dateList[i]
            # scalar outputs
            out[0,i] = ds.albedo.sel(date=date_i).mean().values  # mean albedo across whole tile
            out[1,i] = ds.albedo.sel(date=date_i).std().values # std albedo across whole tile

            # outputs by class
            for j in range(7):

                outClass[0,i,j] = len(ds.classified.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # count instances of each class
                outClass[1,i,j] = np.mean(ds.classified.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # mean of albedo in each class
                outClass[2,i,j] = np.std(ds.classified.sel(date=date_i).values[ds.classified.sel(date=date_i).values==j]) # std of albedo in each class

        outTileXR = xr.DataArray(out,dims=('var','date'),coords={'var':['meanAlbedo','STD Albedo'],'date':dateList})
        outClassXR = xr.DataArray(outClass,dims=('var','date','classID'),coords={'var': ['ClassCount','AlbedoMean','AlbedoSTD'], 'date':dateList,
        'classID':['NONE','SN', 'WAT', 'CC', 'CI', 'LA', 'HA']})

        outTileXR.to_netcdf(savepath + 'OutData_{}_{}.nc'.format(tile,year))
        outClassXR.to_netcdf(savepath + 'OutData_{}_{}_byClass.nc'.format(tile,year))

    # if toggled, remove individual files from /outputs leaving only the concatenated multi-date file
    if config.get('options','remove_individual_files')=='True':

        file_list = glob.glob(outPath+'*Albedo_Data.nc')

        for file in file_list:
            os.remove(file)
            

    if config.get('options','upload_to_blob')=='True':

        # send dataset to azure blob storage and delete from local storage
        azure_cred = configparser.ConfigParser()
        azure_cred.read_file(open(os.environ['AZURE_SECRET']))
        azure = sentinel2_azure.AzureAccess(azure_cred.get('account', 'user'),
                                        azure_cred.get('account', 'key'))
        azure.dataset_to_blob(str(os.environ['PROCESS_DIR']+'outputs/'+ tile + '/'), delete_local_nc=True)

    return


def cloud_interpolator(dataset):
    
    """
    In this cloud interpolator, the missing values are simply filled with the median value for that layer

    """

    # define list of layers depending upon whether snicar
    if config.get('options','retrieve_snicar_params')=='True':

        layers = [dataset.classified, dataset.albedo, dataset.grain_size, dataset.density, dataset.dust, dataset.algae]
    
    else:
    
        layers = [dataset.classified, dataset.albedo]
        
    # define mask
    mask = dataset.FinalMask

    # loop through layers
    for i in range(len(layers)):

        layer = layers[i]
        layer_median = layer.median().values # calculate median value
        layer =layer.where(layer>=0,layer_median) # replace nanswith median

        if i == 0:
            dataset.classified.values = layer
            layer = None

        elif i == 1:
            dataset.albedo.values = layer
            layer = None

        elif i == 2:
            dataset.grain_size.values = layer
            layer = None

        elif i == 3:
            dataset.density.values = layer
            layer = None

        elif i == 4:
            dataset.dust.values = layer
            layer = None
        
        elif i == 5:
            dataset.algae.values = layer
            layer = None

        else:
            print ("ERROR IN PIXELWISE CLOUD INTERPOLATION: counter out of range")

    dataset = dataset.where(mask > 0)

    return dataset