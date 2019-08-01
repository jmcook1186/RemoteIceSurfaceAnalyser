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

class SurfaceClassifier:


    # Bands to use in classifier
    s2_bands_use = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

    # Level 2A scaling factor
    l2a_scaling_factor = 10000



    def load_img_to_xr(self, img_path, resolution):
        """
        Load all bands of image (specified by s2_bands_use) into an xarray
        Dataset, include Icemask and Cloudmask, return the in-memory Dataset.
        Applies scaling factor.

        """
        
        # Open link to each image
        store = []
        for band in self.s2_bands_use:
            fn = glob.glob('%s*%s_%sm.jp2' %(img_path,band,resolution))
            da_band = xr.open_rasterio(fn, chunks={'x': 2000, 'y': 2000})
            # Apply scaling factor
            da_band = da_band / l2a_scaling_factor
            da_band['band'] = [band]
            store.append(da_band)
        
        # Concatenate all bands into a single DataArray
        da = xr.concat(store, dim='band')
        # Rename band dimension for compliance with IceSurfClassifier
        da = da.rename('band', 'b')

        # Create complete dataset
        ds = xr.Dataset({ 'Data': (('b','y','x'), da),
                          'Icemask': (('y', 'x'), Icemask),
                          'Cloudmask': (('x', 'y'), Cloudmask) })

        return ds




    def classify_images(self, pickle_path, S2vals, savepath, tile, date, 
        netcdf_metadata, savefigs=True):

        """

        function applies loaded classifier and a narrowband to broadband albedo conversion to multispectral S2 image saved as
        NetCDF, saving plot and summary data to output folder.

        :param pickle_path: path to trained classifier saved in a pickle file
        :param S2vals: dataset of Sentinel-2 L2A data, probably loaded by load_img_to_xr
        :param savepath: path to output folder
        :param tile: tile ID
        :param date: date of acquisition
        :param netcdf_metadata: dict (key:value) of netcdf metadata, at a minimum author and description
        :param savefigs: Boolean to control whether figure is saved to file
        :return: None

        """
        
        clf = joblib.load(pickle_path)

        # stack the values into a 1D array
        stacked = S2vals.Data.stack(allpoints=['y', 'x'])

        # Transpose and rename so that DataArray has exactly the same layout/labels as the training DataArray.
        # mask out nan areas not masked out by GIMP
        stackedT = stacked.T
        stackedT = stackedT.rename({'allpoints': 'samples'})

        # NEED TO TAKE CARE HERE - DOES DATA DEFINITELY HAVE SAME BAND LABELLING AS TRAINING DATAARRAY?
        # apply classifier
        predicted = clf.predict(stackedT)

        # Unstack back to x,y grid
        predicted = predicted.unstack(dim='samples')

        return predicted



    def calculate_albedo(S2vals):

        # calculate albedo using Liang et al (2002) equation
        albedo = (  0.356 * S2vals.Data.sel(b='B2')  \
                  + 0.130 * S2vals.Data.sel(b='B4')   \
                  + 0.373 * S2vals.Data.sel(b='B8')  \
                  + 0.085 * S2vals.Data.sel(b='B8A') \
                  + 0.072 * S2vals.Data.sel(b='B11') \
                  - 0.0018 )

        return albedo



    def combine_masks():
        
        # update mask so that both GIMP mask and areas not sampled by S2 but 
        # not masked by GIMP both = 0
        mask2 = (S2vals.Icemask.values == 1)  \
              & (concat.sum(dim='b') > 0) \
              & (S2vals.Cloudmask.values == 0)

        return mask2





        ## Collate predicted map, albedo map and projection info into xarray dataset
        
        # 1) Retrieve projection info from S2 datafile and add to netcdf
        srs = osr.SpatialReference()
        srs.ImportFromProj4(S2vals.Data.attrs['crs']) 
        proj_info = xr.DataArray(0, encoding={'dtype': np.dtype('int8')})
        proj_info.attrs['projected_crs_name'] = srs.GetAttrValue('projcs')
        # grid_mapping_name
        proj_info.attrs['grid_mapping_name'] = 'universal_transverse_mercator'
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

        coords_geo = {'y': S2vals.y, 'x': S2vals.x}

        lon_array = xr.DataArray(lon, coords=coords_geo, dims=['y', 'x'],
                                 encoding={'_FillValue': -9999., 'dtype': 'int16', 'scale_factor': 0.000000001})
        lon_array.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']
        lon_array.attrs['units'] = 'degrees'
        lon_array.attrs['standard_name'] = 'longitude'

        lat_array = xr.DataArray(lat, coords=coords_geo, dims=['y', 'x'],
                                 encoding={'_FillValue': -9999., 'dtype': 'int16', 'scale_factor': 0.000000001})
        lat_array.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']
        lat_array.attrs['units'] = 'degrees'
        lat_array.attrs['standard_name'] = 'latitude'

        # 3) add predicted map array and add metadata
        predicted = predicted.fillna(0)
        predicted = predicted.where(mask2 > 0)
        predicted.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
        predicted.name = 'Surface Class'
        predicted.attrs['long_name'] = 'Surface classified using Random Forest'
        predicted.attrs['units'] = 'None'
        predicted.attrs['key'] = netcdf_metadata['predicted_legend']
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
            'Icemask': (['x', 'y'], S2vals.Icemask),
            'Cloudmask':(['x','y'], S2vals.Cloudmask),
            'FinalMask':(['x','y'],mask2),
            proj_info.attrs['grid_mapping_name']: proj_info,
            'longitude': (['x', 'y'], lon_array),
            'latitude': (['x', 'y'], lat_array)
        })

        # add metadata for dataset
        dataset.attrs['Conventions'] = 'CF-1.4'
        dataset.attrs['Author'] = netcdf_metadata['author']
        dataset.attrs['title'] = netcdf_metadata['title']

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



    def albedo_report(self, masterDF, tile, date, savepath):

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



    def concat_all_dates(self, savepath, tile):

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









