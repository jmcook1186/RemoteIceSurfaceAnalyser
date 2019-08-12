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
try:
    import joblib
except ImportError:
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



    def load_img_to_xr(self, img_path, resolution, Icemask, Cloudmask):
        """ Load Sentinel-2 JP2000s into xr.Dataset.

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




    def classify_image(self, pickle_path, S2vals, savepath, tile, date, 
        savefigs=True):
        """ Apply classifier to image.

        function applies pickled classifier to multispectral S2 image saved as
        NetCDF, saving plot and summary data to output folder.

        :param pickle_path: path to trained classifier saved in a pickle file
        :param S2vals: dataset of Sentinel-2 L2A data, probably loaded by load_img_to_xr
        :param savepath: path to output folder
        :param tile: tile ID
        :param date: date of acquisition
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



    def calculate_albedo(self, S2vals):
        """ Calculate albedo using Liang et al (2002) equation """
        
        albedo = (  0.356 * S2vals.Data.sel(b='B2')  \
                  + 0.130 * S2vals.Data.sel(b='B4')   \
                  + 0.373 * S2vals.Data.sel(b='B8')  \
                  + 0.085 * S2vals.Data.sel(b='B8A') \
                  + 0.072 * S2vals.Data.sel(b='B11') \
                  - 0.0018 )

        return albedo



    def combine_masks(self, S2vals):
        """ Combine ice mask and cloud masks """

        mask2 = (S2vals.Icemask.values == 1)  \
              & (S2vals.Data.sum(dim='b') > 0) \
              & (S2vals.Cloudmask.values == 0)

        return mask2



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

