"""
This script pulls images relating to specific Sentinel-2 tiles on specific dates from Azure blob storage and classifies
them using a random forest classifier trained on field spectroscopy data (see github.com/jmcook1186/IceSurfClassifiers).
There is a sequence of quality control functions that determine whether the downloaded image is of sufficient quality to
be used in the analysis or alternatively whether it should be discarded. Reasons for discarding include cloud cover, NaNs
and insufficient ice relative to land or ocean in the image. The sensitivity to these factors is tuned by the user.

Code runs in environment IceSurfClassifiers:
conda create -n IceSurfClassifiers python=3.6 numpy matplotlib scikit-learn seaborn azure rasterio gdal pandas
conda install -c conda-forge xarray georaster sklearn_xarray

"""


# TODO: improve cloud masking algorithm - check the sentinel cloudless python package https://github.com/sentinel-hub/sentinel2-cloud-detector
# TODO: consider creating new classifier and interpolating over bad pixels
# TODO: consider infilling cloudy dates with pixelwiselinear fits from good days
# TODO: consider data output formats and useful parameters to include
# TODO: tidy up console logs and refine logs saved to file

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import xarray as xr
from osgeo import gdal, osr
import georaster
import os
import glob
from azure.storage.blob import BlockBlobService

# matplotlib settings: use ggplot style and turn interactive mode off
mpl.style.use('ggplot')
plt.ioff()

######################################################################################
############################ DEFINE FUNCTIONS ########################################
######################################################################################

def download_imgs_by_date(tile, date, img_path, blob_account_name, blob_account_key):

    """
    This function downloads subsets of images stored remotely in Azure blobs. The blob name is identical to the
    ESA tile ID. Inside each blob are images from every overpass made in June, July and August of the given year.
    The files in blob storage are L2A files, meaning the L1C product has been downloaded from Sentinel-Hub and
    processed for atmospheric conditions, spatial resolution etc using the Sen2cor command line tool.
    This function searches for the blob associated with "tile" and then filteres out a subset according to the prescribed
    date.
    A flags can be raised in this function. The script checks that the correct number of image files have been
    downloaded and that one of them is the cloud mask. If not, the flag is printed to the console and the files
    associated with that particular date for that tile are discarded. The tile and date info are appended to a list of
    failed downloads.
    :param tile: tile ID
    :param date: date of overpass
    :param img_path: path to folder where images and other temp files will be stored
    :param blob_account_name: account name for azure storage account where blobs are stored
    :param blob account key: accesskey for blob storage account
    :return filtered_blob_list: list of files to download
    :return download_flag: Boolean, if True then problem with download, files skipped
    """

    # define blob access
    block_blob_service = BlockBlobService(account_name = blob_account_name, account_key= blob_account_key)

    # setup list
    bloblist = []
    download_flag = False
    QCflag = False

    # append names of all blobs to bloblist
    generator = block_blob_service.list_blobs(tile)
    for blob in generator:
        bloblist.append(blob.name)

    # filter total bloblist to just jp2s, then just for the specified date
    filtered_by_type = [string for string in bloblist if '_20m.jp2' in string]
    filtered_bloblist = [string for string in filtered_by_type if str("L2A_" + date) in string]


    # download the files in the filtered list
    for i in filtered_bloblist:
        print(i)
        try:
            block_blob_service.get_blob_to_path(tile,
                                         i, str(img_path+i[-38:-4]+'.jp2'))
        except:
            print("download failed {}".format(i))

        # index to -38 because this is the filename without paths to folders etc

    # Check downloaded files to make sure all bands plus the cloud mask are present in the wdir
    # Raises download flag (Boolean true) and reports to console if there is a problem

    if len(glob.glob(str(img_path + '*_B*_20m.jp2'))) < 9 or len(glob.glob(str(img_path + '*CLD*_20m.jp2'))) == 0:
        download_flag = True

        print("\n *** DOWNLOAD QC FLAG RAISED *** \n *** There may have been no overpass on this date, or there is a"
              " band image or cloud layer missing from the downloaded directory ***")

    else:
        download_flag = False
        print("\n *** NO DOWNLOAD QC FLAG RAISED: ALL NECESSARY FILES AVAILABLE IN WDIR ***")

    # relevant files now downloaded from blob and stored in the savepath folder

    return filtered_bloblist, download_flag


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


def img_quality_control(Icemask, Cloudmask, minimum_useable_area):

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



def load_model_and_images(img_path, pickle_path, Icemask, Cloudmask):

    """
    function loads classifier from file and loads S2 bands into xarray dataset and saves to NetCDF
    :param img_path: path to S2 image files
    :param pickle_path: path to trained classifier
    :param Icemask: Boolean to mask out non-ice areas
    :param Cloudmask: Boolean to mask out cloudy pixels
    :return: clf: classifier loaded in from .pkl file;
    """
    # Sentinel 2 dataset
    # create xarray dataset with all bands loaded from jp2s. Values are reflectance.
    fileB2 = glob.glob(str(img_path + '*B02_20m.jp2'))
    fileB3 = glob.glob(str(img_path + '*B03_20m.jp2'))
    fileB4 = glob.glob(str(img_path + '*B04_20m.jp2'))
    fileB5 = glob.glob(str(img_path + '*B05_20m.jp2'))
    fileB6 = glob.glob(str(img_path + '*B06_20m.jp2'))
    fileB7 = glob.glob(str(img_path + '*B07_20m.jp2'))
    fileB8 = glob.glob(str(img_path + '*B8A_20m.jp2'))
    fileB11 = glob.glob(str(img_path + '*B11_20m.jp2'))
    fileB12 = glob.glob(str(img_path + '*B12_20m.jp2'))

    daB2 = xr.open_rasterio(fileB2[0], chunks={'x': 2000, 'y': 2000})
    daB3 = xr.open_rasterio(fileB3[0], chunks={'x': 2000, 'y': 2000})
    daB4 = xr.open_rasterio(fileB4[0], chunks={'x': 2000, 'y': 2000})
    daB5 = xr.open_rasterio(fileB5[0], chunks={'x': 2000, 'y': 2000})
    daB6 = xr.open_rasterio(fileB6[0], chunks={'x': 2000, 'y': 2000})
    daB7 = xr.open_rasterio(fileB7[0], chunks={'x': 2000, 'y': 2000})
    daB8 = xr.open_rasterio(fileB8[0], chunks={'x': 2000, 'y': 2000})
    daB11 = xr.open_rasterio(fileB11[0], chunks={'x': 2000, 'y': 2000})
    daB12 = xr.open_rasterio(fileB12[0], chunks={'x': 2000, 'y': 2000})

    daB2 = xr.DataArray.squeeze(daB2, dim='band')
    daB3 = xr.DataArray.squeeze(daB3, dim='band')
    daB4 = xr.DataArray.squeeze(daB4, dim='band')
    daB5 = xr.DataArray.squeeze(daB5, dim='band')
    daB6 = xr.DataArray.squeeze(daB6, dim='band')
    daB7 = xr.DataArray.squeeze(daB7, dim='band')
    daB8 = xr.DataArray.squeeze(daB8, dim='band')
    daB11 = xr.DataArray.squeeze(daB11, dim='band')
    daB12 = xr.DataArray.squeeze(daB12, dim='band')

    S2vals = xr.Dataset({'B02': (('y', 'x'), daB2.values / 10000), 'B03': (('y', 'x'), daB3.values / 10000),
                         'B04': (('y', 'x'), daB4.values / 10000), 'B05': (('y', 'x'), daB5.values / 10000),
                         'B06': (('y', 'x'), daB6.values / 10000), 'B07': (('y', 'x'), daB7.values / 10000),
                         'B08': (('y', 'x'), daB8.values / 10000), 'B11': (('y', 'x'), daB11.values / 10000),
                         'B12': (('y', 'x'), daB12.values / 10000), 'Icemask': (('y', 'x'), Icemask),
                         'Cloudmask': (('x', 'y'), Cloudmask)})

    S2vals.to_netcdf(img_path + "S2vals.nc", mode='w')

    # flush disk
    S2vals = None
    daB2 = None
    daB3 = None
    daB4 = None
    daB5 = None
    daB6 = None
    daB7 = None
    daB8 = None
    daB11 = None
    daB12 = None
    Cloudmask = None
    Icemask = None

    #load pickled model
    clf = joblib.load(pickle_path)

    return clf


def ClassifyImages(clf, img_path, savepath, tile, date, savefigs=True):

    """

    function applies loaded classifier and a narrowband to broadband albedo conversion to multispectral S2 image saved as
    NetCDF, saving plot and summary data to output folder.
    :param clf: trained classifier loaded from file
    :param img_path: path to S2 images
    :param savepath: path to output folder
    :param tile: tile ID
    :param date: date of acquisition
    :param savefigs: Boolean to control whether figure is saved to file
    :return: None

    """

    with xr.open_dataset(img_path + "S2vals.nc",chunks={'x':2000,'y':2000}) as S2vals:
        # Set index for reducing data
        band_idx = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9], name='bands')

        # concatenate the bands into a single dimension ('bands_idx') in the data array
        concat = xr.concat([S2vals.B02, S2vals.B03, S2vals.B04, S2vals.B05, S2vals.B06, S2vals.B07,
                            S2vals.B08, S2vals.B11, S2vals.B12], band_idx)

        # stack the values into a 1D array
        stacked = concat.stack(allpoints=['y', 'x'])

        # Transpose and rename so that DataArray has exactly the same layout/labels as the training DataArray.
        # mask out nan areas not masked out by GIMP
        stackedT = stacked.T
        stackedT = stackedT.rename({'allpoints': 'samples'})

        # apply classifier
        predicted = clf.predict(stackedT)

        # Unstack back to x,y grid
        predicted = predicted.unstack(dim='samples')

        #calculate albeod using Liang et al (2002) equation
        albedo = xr.DataArray(0.356 * (concat.values[1]) + 0.13 * (concat.values[3]) + 0.373 * \
                       (concat.values[6]) + 0.085 * (concat.values[7]) + 0.072 * (concat.values[8]) - 0.0018)

        #update mask so that both GIMP mask and areas not sampled by S2 but not masked by GIMP both = 0
        mask2 = (S2vals.Icemask.values ==1) & (concat.sum(dim='bands')>0) & (S2vals.Cloudmask.values == 0)

        # collate predicted map, albedo map and projection info into xarray dataset
        # 1) Retrieve projection info from S2 datafile and add to netcdf
        srs = osr.SpatialReference()
        srs.ImportFromProj4('+init=epsg:32622') # Get info for UTM zone 22N
        proj_info = xr.DataArray(0, encoding={'dtype': np.dtype('int8')})
        proj_info.attrs['projected_crs_name'] = srs.GetAttrValue('projcs')
        proj_info.attrs['grid_mapping_name'] = 'UTM'
        proj_info.attrs['scale_factor_at_central_origin'] = srs.GetProjParm('scale_factor')
        proj_info.attrs['standard_parallel'] = srs.GetProjParm('latitude_of_origin')
        proj_info.attrs['straight_vertical_longitude_from_pole'] = srs.GetProjParm('central_meridian')
        proj_info.attrs['false_easting'] = srs.GetProjParm('false_easting')
        proj_info.attrs['false_northing'] = srs.GetProjParm('false_northing')
        proj_info.attrs['latitude_of_projection_origin'] = srs.GetProjParm('latitude_of_origin')

        # 2) Create associated lat/lon coordinates DataArrays using georaster (imports geo metadata without loading img)
        # see georaster docs at https:/media.readthedocs.org/pdf/georaster/latest/georaster.pdf

        # find B02 jp2 file
        fileB2 = glob.glob(str(img_path + '*B02_20m.jp2'))
        fileB2 = fileB2[0]

        S2 = georaster.SingleBandRaster(fileB2, load_data=False)
        lon, lat = S2.coordinates(latlon=True)
        S2 = None

        S2 = xr.open_rasterio(fileB2, chunks={'x': 2000, 'y': 2000})
        coords_geo = {'y': S2['y'], 'x': S2['x']}
        S2 = None

        lon_array = xr.DataArray(lon, coords=coords_geo, dims=['y', 'x'],
                                 encoding={'_FillValue': -9999., 'dtype': 'int16', 'scale_factor': 0.000000001})
        lon_array.attrs['grid_mapping'] = 'UTM'
        lon_array.attrs['units'] = 'degrees'
        lon_array.attrs['standard_name'] = 'longitude'

        lat_array = xr.DataArray(lat, coords=coords_geo, dims=['y', 'x'],
                                 encoding={'_FillValue': -9999., 'dtype': 'int16', 'scale_factor': 0.000000001})
        lat_array.attrs['grid_mapping'] = 'UTM'
        lat_array.attrs['units'] = 'degrees'
        lat_array.attrs['standard_name'] = 'latitude'

        # 3) add predicted map array and add metadata
        predictedxr = xr.DataArray(predicted.values, coords=coords_geo, dims=['y', 'x'])
        predictedxr = predictedxr.fillna(0)
        predictedxr = predictedxr.where(mask2>0)
        predictedxr.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
        predictedxr.name = 'Surface Class'
        predictedxr.attrs['long_name'] = 'Surface classified using Random Forest'
        predictedxr.attrs['units'] = 'None'
        predictedxr.attrs[
            'key'] = 'Snow:1; Water:2; Cryoconite:3; Clean Ice:4; Light Algae:5; Heavy Algae:6'
        predictedxr.attrs['grid_mapping'] = 'UTM 22N'

        # add albedo map array and add metadata
        albedoxr = xr.DataArray(albedo.values, coords=coords_geo, dims=['y', 'x'])
        albedoxr = albedoxr.fillna(0)
        albedoxr = albedoxr.where(mask2 > 0)
        albedoxr.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
        albedoxr.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
        albedoxr.attrs['units'] = 'dimensionless'
        albedoxr.attrs['grid_mapping'] = 'UTM 22N'

        # collate data arrays into a dataset
        dataset = xr.Dataset({

            'classified': (['x', 'y'], predictedxr),
            'albedo':(['x','y'],albedoxr),
            'Icemask': (['x', 'y'], S2vals.Icemask.values),
            'Cloudmask':(['x','y'], S2vals.Cloudmask.values),
            'FinalMask':(['x','y'],mask2),
            'Projection': proj_info,
            'longitude': (['x', 'y'], lon_array),
            'latitude': (['x', 'y'], lat_array)
        })

        # add metadata for dataset
        dataset.attrs['Conventions'] = 'CF-1.4'
        dataset.attrs['Author'] = 'Joseph Cook (University of Sheffield, UK)'
        dataset.attrs[
            'title'] = 'Classified surface and albedo maps produced from Sentinel-2 ' \
                       'imagery of the SW Greenland Ice Sheet'

        # Additional geo-referencing
        dataset.attrs['nx'] = len(dataset.x)
        dataset.attrs['ny'] = len(dataset.y)
        dataset.attrs['xmin'] = float(dataset.x.min())
        dataset.attrs['ymax'] = float(dataset.y.max())
        dataset.attrs['spacing'] = 20

        # NC conventions metadata for dimensions variables
        dataset.x.attrs['units'] = 'meters'
        dataset.x.attrs['standard_name'] = 'projection_x_coordinate'
        dataset.x.attrs['point_spacing'] = 'even'
        dataset.x.attrs['axis'] = 'x'

        dataset.y.attrs['units'] = 'meters'
        dataset.y.attrs['standard_name'] = 'projection_y_coordinate'
        dataset.y.attrs['point_spacing'] = 'even'
        dataset.y.attrs['axis'] = 'y'

        dataset.to_netcdf(savepath + "{}_{}_Classification_and_Albedo_Data.nc".format(tile,date), mode='w')

        dataset = None

    if savefigs:
        # set colormap and background colours
        cmap1 = mpl.colors.ListedColormap(
            ['white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
        cmap1.set_under(color='white')  # make sure background is white
        cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
        cmap2.set_under(color='white')  # make sure background is white

        # initialize figure object, add title and set class labels
        fig = plt.figure(figsize=(6, 8))
        plt.title("Classified ice surface and its albedos from UAV imagery: SW Greenland Ice Sheet", fontsize=28)
        class_labels = ['Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae']

        # first subplot = classified map
        ax1 = plt.subplot(211)
        img = predictedxr.plot(cmap=cmap1, add_colorbar=False)
        cbar = fig.colorbar(mappable=img, ax=ax1)

        # workaround to get colorbar labels centrally positioned
        tick_locs = np.arange(1, len(class_labels), 0.92)
        cbar.set_ticks(tick_locs)
        cbar.ax.set_yticklabels(class_labels, rotation=45, va='center')
        plt.title('Classified Surface Map (UTM coordinates)'), ax1.set_aspect('equal')

        # second subplot = albedo map
        ax2 = plt.subplot(212)
        albedoxr.plot(cmap=cmap2, vmin=0, vmax=1, ax=ax2), plt.title('Albedo Map (UTM coordinates)')
        ax2.set_aspect('equal')

        plt.savefig(str(savepath + "{}_{}_Classified_Albedo_Map.png".format(tile, date)), dpi=300)
        plt.close()

    return


def albedo_report(masterDF, tile, date, savepath):

    with xr.open_dataset(savepath + "{}_{}_Classification_and_Albedo_Data.nc".format(tile, date),
                         chunks={'x': 2000, 'y': 2000}) as dataset:

        predicted = np.array(dataset.classified.values).ravel()
        albedo = np.array(dataset.albedo.values).ravel()

        albedoDF = pd.DataFrame()
        albedoDF['pred'] = predicted
        albedoDF['albedo'] = albedo
        albedoDF['tile'] = tile
        albedoDF['date'] = date

        countDF = albedoDF.groupby(['pred']).count()
        summaryDF = albedoDF.groupby(['pred']).describe()['albedo']

        ####################################

        summaryxr = xr.DataArray(summaryDF, dims=('classID', 'metric'),
                                 coords={'classID': ['SN', 'WAT', 'CC', 'CI', 'LA', 'HA'],
                                         'metric': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
                                         }, attrs={'date': date})

        summaryxr.to_netcdf(str(savepath+'summary_data_{}_{}.nc'.format(tile, date)))

        algal_coverage = (sum(summaryxr.sel(classID=['HA', 'LA'], metric='count')) / (
            sum(summaryxr.sel(classID=['HA','LA','WAT', 'CC', 'CI'], metric='count').values))) * 100

        #####################################

        masterDF = masterDF.append(albedoDF, ignore_index=True)

    return summaryDF, masterDF


def concat_all_dates(savepath, tile):

    """
    Function concatenates xarrays from individual dates into one master dataset for each tile.
    The dimensions are: date, classID and metric
    The coordinates on each dimension are accessed by their index in the following lists:
    date [0: 1st date, 1: 2nd date, 2: 3rd date]
    classID [0: SN, 1: WAT , 2: CC, 3: CI, 4: LA, 5: HA]
    metric [0: count, 1: mean, 2: std, 3: min, 4: 25% , 5: 50% , 6: 75%, 7: max ]
    The order of the indexes is: concat_data[date, ID, metric]
    so to access the total number of pixels classed as snow on the first date:
    >> concat_dataset[0,0,0].values
    to access the mean albedo of cryoconite covered pixels on all dates:
    >> concat_dataset[:,2,1].values
    or alternatively use xr.sel() which allows indexing by label, e.g.:
    concat_dataset.sel(classID = 'HA', metric = 'mean', date = '20160623')
    :param masterDF:
    :return:
    """

    data = []
    ds = []
    dates = []

    xrlist = glob.glob(str(savepath + 'summary_data_' + '*.nc')) # find all summary datasets

    for i in np.arange(0,len(xrlist),1):
        ds = xr.open_dataarray(xrlist[i])
        data.append(ds)
        date = ds.date
        dates.append(date)

        try:
            concat_data = xr.concat(data, dim=pd.Index(dates, name='date'))

            savefilename = str(savepath+'summary_data_all_dates_{}.nc'.format(tile))
            concat_data.to_netcdf(savefilename,'w')
            concat_data = None  # flush

        except:
            print("could not concatenate output files - there is probably only one output file available")

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


###################################################################################
######## DEFINE BLOB ACCESS, GLOBAL VARIABLES AND HARD-CODED PATHS TO FILES #######
###################################################################################

blob_account_name = 'tothepoles'
blob_account_key = 'HwYM3ZVtNv3j14/3iF57Zb9qIA7O5DTcB9Xx7pEoG1Ctw0fqJ7W5/JMSxfzKwp5tULqYVqH42dbKigvRg2QJqw=='
img_path = '/home/joe/Desktop/blobtest/'
pickle_path = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Sentinel2_classifier.pkl'
Icemask_in = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/merged_mask.tif'
Icemask_out = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/GIMP_MASK.nc'
cloudProbThreshold = 10 # % probability threshold for classifying an individual pixel and cloudy or not cloudy (>threshold = discard)
minimum_useable_area = 30 # minimum proportion of total image comprising useable pixels. < threshold = image discarded

# set up empty lists and dataframes to append to
download_problem_list =[] # empty list to append details of skipped tiles due to missing info
QC_reject_list = [] # empty list to append details of skipped tiles due to cloud cover
good_tile_list = [] # empty list to append tiles used in analysis
masterDF = pd.DataFrame()
dates = []

###################################################################################
######################### SET TILE AND DATE RANGE #################################
###################################################################################

# list all tiles to include
tiles = ['22WEV']#['22XDF', '21XWD', '21XWC', '22WEV', '22WEU', '22WEA', '22WEB', '22WED', '21XWB', '21XXA', '21WEC',
         #'21XVD','22WES', '22VER', '22WET']

# specify year and month - currently automatically includes all days in each month (i.e. days not user defined).
years = [2016,2017,2018] # specify years
months = [6,7,8] # specify months of the year

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


for tile in tiles:

    tile = tile.lower() #azure blob store is case sensitive: force lower case
    #first create directory to save outputs to
    dirName = str(img_path+'outputs/'+tile+"/")

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
        filtered_bloblist, download_flag = download_imgs_by_date(tile = tile, date = date, img_path = img_path,
                                            blob_account_name = blob_account_name, blob_account_key = blob_account_key)

        # check download and only proceed if correct no. of files and cloud layer present
        if download_flag == False:

            print("\n*** No Download flag raised *** \n *** Checking cloud, ice and NaN cover ***")

            Icemask, Cloudmask = format_mask(img_path, Icemask_in, Icemask_out, cloudProbThreshold)
            QCflag, useable_area = img_quality_control(Icemask, Cloudmask, minimum_useable_area)

            # Check image is not too cloudy. If OK, proceed, if not, skip tile/date
            if QCflag == False:

                print("\n *** No cloud or ice cover flags: proceeding with image analysis for tile {}".format(tile))
                good_tile_list.append('{}_{}_useable_area = {} '.format(tile,date,useable_area))

                try: # use try/except so that any images that slips through QC and then fail do not break run
                    clf = load_model_and_images(img_path, pickle_path, Icemask, Cloudmask)
                    ClassifyImages(clf, img_path, savepath, tile, date, savefigs=True)
                    summaryDF, masterDF = albedo_report(masterDF, tile, date, savepath)

                except:
                    print("\n *** IMAGE ANALYSIS ATTEMPTED AND FAILED FOR {} {}: MOVING ON TO NEXT DATE \n".format(tile,date))

            else:
                print("\n *** QC Flag Raised *** \n*** Skipping tile {} on {} due to QCflag: {} % useable pixels ***".format(tile, date, np.round(useable_area,4)))
                QC_reject_list.append('{}_{}_useable_area = {} '.format(tile,date,useable_area))

        else:
            print("\n*** Download Flag Raised ***\n *** Skipping tile {} on {} due to download flag ***".format(tile, date))
            download_problem_list.append('{}_{}'.format(tile,date))

        clear_img_directory(img_path)

    print("\n *** COLLATING INDIVIDUAL TILES INTO FINAL DATASET***")
    concat_dataset = concat_all_dates(savepath, tile)

    # save logs to csv files
    print("\n *** SAVING QC LOGS TO TXT FILES ***")
    print("\n *** aborted_downloads.txt, rejected_by_qc.txt and good_tiles.txt saved to output folder ")
    np.savetxt(str(savepath+"/aborted_downloads.csv"), download_problem_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath+"/rejected_by_qc.csv"), QC_reject_list, delimiter=",", fmt='%s')
    np.savetxt(str(savepath+"/good_tiles.csv"), good_tile_list, delimiter=",", fmt='%s')

print("\n ************************\n *** COMPLETED RUN  *** \n *********************")

