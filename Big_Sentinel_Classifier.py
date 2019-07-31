"""
Functions for classifying Sentinel 2 images using a trained classification model

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

        dataset=None

    if savefigs:

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









