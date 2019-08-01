"""
Functions for local-system management of Sentinel-2 files.

"""

from collections import OrderedDict
import os
import glob
import shutil
import numpy as np


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
    print("*** CHECKING DATA QUALITY ***")

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
    print("{} % of the image is composed of useable pixels".format(np.round(useable_area,2)))
    print("*** FINISHED QUALITY CONTROL ***")

    # raise flags
    if (useable_area < minimum_useable_area):

        QCflag = True
        print("*** THE NUMBER OF USEABLE PIXELS IS LESS THAN THE MINIMUM THRESHOLD: DISCARDING IMAGE *** \n")

    else:

        QCflag = False
        print("*** SUFFICIENT GOOD PIXELS: PROCEEDING WITH IMAGE ANALYSIS ***\n")

    return QCflag, useable_area