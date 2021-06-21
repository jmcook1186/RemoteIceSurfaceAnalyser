"""
Functions for classifying Sentinel 2 images using a trained classification model

"""

import numpy as np
import pandas as pd
import xarray as xr
import ebmodel as ebm
import ebmodel as ebm
import glob
import os
import dask

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

class SurfaceClassifier:

    # Bands to use in classifier
    s2_bands_use = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

    # Level 2A scaling factor
    l2a_scaling_factor = 10000

    # names of bands as used in classifier
    NAME_bands = 'bands'
    NAME_x = 'x'
    NAME_y = 'y'

    def __init__(self, pkl):
        """ Initialise classifier. 

        :param pkl: path of pickled classifier.
        :type pkl: str

        """
        self.classifier = joblib.load(pkl)

        return



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

            if len(fn) > 1:
                raise ValueError("Multiple bands named {} in blob container. One expected.".format(fn))

            else:
                try:
                    fn = fn[0]

                except:
                    raise IndexError("At least one band missing from S2 blob container")        
            
            da_band = xr.open_rasterio(fn, chunks={self.NAME_x: 2000, self.NAME_y: 2000})
            crs = da_band.attrs['crs']
            # Apply scaling factor
            da_band = da_band / self.l2a_scaling_factor
            da_band[self.NAME_bands] = band
            store.append(da_band)
        
        # Concatenate all bands into a single DataArray
        da = xr.concat(store, dim=self.NAME_bands).squeeze()

        # Create complete dataset
        ds = xr.Dataset({ 'Data': ((self.NAME_bands,self.NAME_y,self.NAME_x), da),
                          'Icemask': ((self.NAME_y,self.NAME_x), Icemask),
                          'Cloudmask': ((self.NAME_y,self.NAME_x), Cloudmask) },
                          coords={self.NAME_bands:self.s2_bands_use, 
                                  self.NAME_y:da_band[self.NAME_y],
                                  self.NAME_x:da_band[self.NAME_x]})

        ds.Data.attrs['crs'] = crs

        return ds


    def classify_image(self, S2vals, savepath, tile, date, 
        savefigs=True):
        """ Apply classifier to image.

        function applies pickled classifier to multispectral S2 image saved as
        NetCDF, saving plot and summary data to output folder.
        """

        # stack the values into a 1D array
        stacked = S2vals.Data.stack(allpoints=[self.NAME_y,self.NAME_x])

        # Transpose and rename so that DataArray has exactly the same layout/labels as the training DataArray.
        # mask out nan areas not masked out by GIMP
        stackedT = stacked.T
        stackedT = stackedT.rename({'allpoints': 'samples'})

        # apply classifier
        predicted = self.classifier.predict(stackedT)

        # Unstack back to x,y grid
        predicted = predicted.unstack(dim='samples')

        return predicted


    def calculate_albedo(self, S2vals):
        """ Calculate albedo using Liang et al (2002) equation

        For some reason B8A is being used in unforked version of repo, even
        though Naegeli uses B08...

        See also Naegeli et al 2017, Remote Sensing
        """
        
        albedo = (  0.356 * S2vals.Data.loc[{self.NAME_bands:'B02'}] \
                  + 0.130 * S2vals.Data.loc[{self.NAME_bands:'B04'}] \
                  + 0.373 * S2vals.Data.loc[{self.NAME_bands:'B8A'}] \
                  + 0.085 * S2vals.Data.loc[{self.NAME_bands:'B11'}] \
                  + 0.072 * S2vals.Data.loc[{self.NAME_bands:'B12'}] \
                  - 0.0018 )

        return albedo
    
    
    def calculate_2DBA(self, S2vals):

        Index2DBA = S2vals.Data.loc[{self.NAME_bands:'B05'}]/S2vals.Data.loc[{self.NAME_bands:'B04'}]
        predict2DBA = 201100* Index2DBA -194500

        return Index2DBA, predict2DBA

    
    def combine_masks(self, S2vals):
        """ Combine ice mask and cloud masks """

        mask2 = (S2vals.Icemask.values == 1)  \
            & (S2vals.Data.sum(dim=self.NAME_bands) > 0) \
            & (S2vals.Cloudmask.values == 0)

        return mask2


    def invert_snicar_multi_LUT(self, S2vals, mask2, predictedxr, predict2DBA, wavelengths, idx, tile, year, month):
        """
        function inverts snicar but with 2BDA band ratio acting as first-pass filter
        i.e. the 2BDA value is used to restrict the LUT to a particular parameter range
        This allows finer resolution of parameters without sacrificing computation time

        """

        # REFORMAT INPUT DATA
        # 1) reshape band data into 1D stack, rename and chunk
        stacked = S2vals.Data.stack(allpoints=[self.NAME_y,self.NAME_x])
        stackedT = stacked.T
        stackedT = stackedT.rename({'allpoints': 'samples'})
        stackedT = stackedT.chunk(20000,9)

        # 2) Repeat for 2BDA data
        stacked2DBA = predict2DBA.stack(allpoints=[self.NAME_y,self.NAME_x])
        stacked2DBA = stacked2DBA.T
        stacked2DBA = stacked2DBA.rename({'allpoints': 'samples'})
        stacked2DBA.chunk(20000)

        dz = [0.02, 0.03, 0.04, 0.05, 0.075 , 0.1]
        densities = [600, 650, 700, 750, 800, 850]
        grain_rds = [600, 700, 800, 900, 1000]
        algae = [0, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000]
        solzens = [40, 45, 50, 55]
        algae_low = [0, 5000, 7500, 10000, 12500]
        algae_mid = [10000, 12500, 15000, 17500]
        algae_high = [15000, 17500, 20000, 25000, 30000]

        # load the LUT associated with the appropriate solar zenith angle
        # then reshape to correct wavelength range and flatten
        LUT = np.load(str(os.environ['PROCESS_DIR'] +'LUT.npy'))

        # select appropriate SZA from file metadata
        files = glob.glob(os.environ['PROCESS_DIR'] + '*.jp2') # grab current files
        file = files[5] #take first file from list
        print(file[52:60]) # print year,month and day from the filename
        rowID = file[52:60] # make the date info from the filename rowID

        # open the metadata file
        metadataDF = pd.read_csv('/datadrive/BigIceSurfClassifier/Process_Dir/AllFiles_coszen.csv')

        #find the row corresponding to the open file
        DF2 = metadataDF[(metadataDF['filename'].str.contains(rowID)) & (metadataDF['filename'].str.contains(tile.upper()))]

        # condition to protect against missing or invalid data in metadata file (defaults to most common SZA)
        if len(DF2)==0:
            coszen = 0.7
        else:
            # grab the coszen value from the metadata
            coszen = DF2['coszen'].values[0]

        coszen = int(coszen*10) #round to nearest tenth

        if coszen >=0.75:
            zen_idx = 0
        elif (coszen >=0.65) & (coszen < 0.75):
            zen_idx = 1
        elif (coszen >=0.58) & (coszen < 0.65):
            zen_idx = 2
        else:
            zen_idx = 3

        print(LUT.shape)
        LUT_low = LUT[zen_idx,:,:,:,0:5,10:].reshape(len(densities)*len(grain_rds)*len(dz)*len(algae_low),len(wavelengths[10:]))
        LUT_mid = LUT[zen_idx,:,:,:,3:7,10:].reshape(len(densities)*len(grain_rds)*len(dz)*len(algae_mid),len(wavelengths[10:]))
        LUT_high = LUT[zen_idx,:,:,:,5:,10:].reshape(len(densities)*len(grain_rds)*len(dz)*len(algae_high),len(wavelengths[10:]))

        # reshape LUT to correct wavelengths and convert to xr.DataArray_hi
        LUT_low = LUT_low[:,idx]
        LUT_low = xr.DataArray(LUT_low,dims=('spectrum','bands')).astype(np.float32)
        LUT_mid = LUT_mid[:,idx]
        LUT_mid = xr.DataArray(LUT_mid,dims=('spectrum','bands')).astype(np.float32)
        LUT_high = LUT_high[:,idx]
        LUT_high = xr.DataArray(LUT_high,dims=('spectrum','bands')).astype(np.float32)

        
        # set up empty dataArrays for calculating error between S2 and LUT
        error_array = np.zeros(shape=stackedT.shape)
        error_array1 = xr.DataArray(error_array,dims=('spectrum','bands')).astype(np.float32)
        error_array2 = xr.DataArray(error_array,dims=('spectrum','bands')).astype(np.float32)
        error_array3 = xr.DataArray(error_array,dims=('spectrum','bands')).astype(np.float32)


        # subtract S2 reflectance from LUT 
        print("subtraction step")    
        error_array1 = LUT_low-stackedT.astype(np.float32)
        error_array2 = LUT_mid-stackedT.astype(np.float32)
        error_array3 = LUT_high-stackedT.astype(np.float32)

        # make absolute (i.e. -ves to +ve)
        print("taking abs")
        idx_array1 = xr.apply_ufunc(abs,error_array1,dask='allowed')
        idx_array2 = xr.apply_ufunc(abs,error_array2,dask='allowed')
        idx_array3 = xr.apply_ufunc(abs,error_array3,dask='allowed')


        # flush used error_arrays from memory
        error_array1 = None
        error_array2 = None
        error_array3 = None


        # take mean of absolute error across wavelength
        print("finding mean")
        idx_array1 = xr.apply_ufunc(np.mean,idx_array1,input_core_dims=[['bands']],dask='allowed',kwargs={'axis':-1}) 
        idx_array2 = xr.apply_ufunc(np.mean,idx_array2,input_core_dims=[['bands']],dask='allowed',kwargs={'axis':-1})
        idx_array3 = xr.apply_ufunc(np.mean,idx_array3,input_core_dims=[['bands']],dask='allowed',kwargs={'axis':-1})
    
        # find index of lowest mean error 
        print("finding argmin")
        idx_array1 = xr.apply_ufunc(np.argmin,idx_array1,input_core_dims=[['spectrum']],dask='allowed',kwargs={'axis':-1})
        idx_array2 = xr.apply_ufunc(np.argmin,idx_array2,input_core_dims=[['spectrum']],dask='allowed',kwargs={'axis':-1})
        idx_array3 = xr.apply_ufunc(np.argmin,idx_array3,input_core_dims=[['spectrum']],dask='allowed',kwargs={'axis':-1})

        print("masking by 2BDA")
        idx_array1 = np.where((stacked2DBA.values>=10000)&(stacked2DBA.values<20000),idx_array2, idx_array1)
        idx_array1 = np.where(stacked2DBA.values>20000,idx_array3, idx_array1)



        param_array1 = np.array(np.unravel_index(idx_array1,[len(densities),len(grain_rds),len(dz),len(algae_low)]))
        param_array2 = np.array(np.unravel_index(idx_array2,[len(densities),len(grain_rds),len(dz),len(algae_mid)]))
        param_array3 = np.array(np.unravel_index(idx_array3,[len(densities),len(grain_rds),len(dz),len(algae_high)]))
        # creat esingle array with param idxs aligned with 2BDA range (i.e use different LUT depending on LUT)
        
        param_array1 = np.where((stacked2DBA.values>=10000)&(stacked2DBA.values<20000), param_array2, param_array1)
        param_array1 = np.where(stacked2DBA.values>=20000, param_array3, param_array1)

        # flush now-redundant param_arrays from memory
        param_array2 = None
        param_array3= None

        # flush no-redundant idx_arrays from memory
        idx_array1 = None
        idx_array2 = None
        idx_array3 = None

        ## Start the bit that hurts my head...
        # use the indexes to retrieve the actual parameter values for each pixel from the LUT indexes
        # only the first value from the parameter arrays is needed.

        param_names = ['density','reff','dz','algae']
        
        # concatenate param values from each LUT into list of lists

        algae = [algae_low, algae_mid, algae_high]
        densities = [densities,densities, densities]
        dz = [dz, dz, dz]
        grain_rds = [grain_rds, grain_rds, grain_rds]
        
        # define params to iterate through - top level iterator
        params = [densities, grain_rds, dz, algae]

        # loop through params (e.g. j == densities)
        for paramChoice in np.arange(0,len(params),1):  
            
            # select parameter from parameter list
            param = params[paramChoice]

            # set up empty result arrays
            result_array = np.empty(shape=stacked2DBA.shape)
            result_array2 = np.empty(shape=stacked2DBA.shape)
            result_array3 = np.empty(shape=stacked2DBA.shape)

            # select array of parameter values from each parameter      
            for paramValues in np.arange(0,len(param),1):
                
                # iterate through individual values in each parameter value list
                for ParamVal in np.arange(0,len(param[paramValues]),1):
                    
                    # condition checks which array of param values we are in
                    # each param value array is specific to an individual LUT
                    # create separate array for each LUT
                    if paramValues==0:
                        result_array[param_array1[paramChoice]==ParamVal] = param[paramValues][ParamVal]

                    elif paramValues==1:
                        result_array2[param_array1[paramChoice]==ParamVal] = param[paramValues][ParamVal]
                    
                    elif paramValues==2:
                        result_array3[param_array1[paramChoice]==ParamVal] = param[paramValues][ParamVal]   
    
                # combine individual result arrays by selecting values from each based on 2BDA value
                result_array = np.where((stacked2DBA.values>=10000)&(stacked2DBA.values<20000),result_array2, result_array)
                result_array = np.where(stacked2DBA >= 20000, result_array3, result_array)

            # reshape result_array to original dims, add metadata, convert to xr DataArray, apply mask
            result_array = result_array.reshape(int(np.sqrt(len(stackedT))),int(np.sqrt(len(stackedT))))
            resultxr = xr.DataArray(data=result_array,dims=['y','x'], coords={'x':S2vals.x, 'y':S2vals.y}).chunk(2000,2000)
            
            # mask to ice area
            resultxr = resultxr.where(mask2 > 0)

            # send to netcdf and flush memory
            resultxr.to_netcdf(str(os.environ['PROCESS_DIR']+ "{}.nc".format(param_names[paramChoice])))
            
            # flush now-redundant result_arrays from memory
            result_array = None
            result_array2 = None


            resultxr = None

        return




    def invert_snicar_single_LUT(self, S2vals, mask2, predictedxr, densities, grain_rds, algae, wavelengths, idx, tile, year, month):
        """
        Pixelwise retrieval of snicar RT params by matching spectra against snicar-generated LUT loaded from process_dir
        """
        
        # stack the values into a 1D array
        stacked = S2vals.Data.stack(allpoints=[self.NAME_y,self.NAME_x])

        # Transpose and rename so that DataArray has exactly the same layout/labels as the training DataArray.
        stackedT = stacked.T
        stackedT = stackedT.rename({'allpoints': 'samples'})

        # RECHUNK STACKEDT: choice of chunk size is crucial for maximising speed while preventing memory over-allocation.
        # While 20000 seems small there are very large intermediate arrays spawned by the compute() function.
        # chunks >~20000 give memory error due to size of intermediate arrays 
        stackedT = stackedT.chunk(20000,9)

        #################################
        # SELECT AND LOAD LUT
        #################################

        # identify coszen and load appropriate lut

        files = glob.glob(os.environ['PROCESS_DIR'] + '*.jp2') # grab current files
        file = files[5] #take first file from list
        print(file[52:60]) # print year,month and day from the filename

        rowID = file[52:60] # make the date info from the filename rowID

        print(rowID)
        print(tile.upper())

        # open the metadata file
        metadataDF = pd.read_csv('/datadrive/BigIceSurfClassifier/Process_Dir/AllFiles_coszen.csv')

        #find the row corresponding to the open file
        DF2 = metadataDF[(metadataDF['filename'].str.contains(rowID)) & (metadataDF['filename'].str.contains(tile.upper()))]

        # condition to protect against missing or invalid data in metadata file (defaults to most common SZA)
        if len(DF2)==0:

            coszen = 0.7

        else:
            # grab the coszen value from the metadata
            coszen = DF2['coszen'].values[0]

        coszen = int(coszen*10) #round to nearest tenth

        # load the LUT associated with the appropriate solar zenith angle
        LUT = np.load(str(os.environ['PROCESS_DIR'] +'LUT.npy'.format(coszen)))

        # reformat LUT: flatten LUT from 3D to 2D array with one column per combination
        # of RT params, one row per wavelength

        LUT = LUT.reshape(len(densities)*len(grain_rds)*len(algae),len(wavelengths))

        ##################################
        ###################################

        # find most similar LUT spectrum for each pixel in S2 image
        # astype(float 16) to reduce memory allocation (default was float64)
        LUT = LUT[:,idx] # reduce wavelengths to only the 9 that match the S2 image

        LUT = xr.DataArray(LUT,dims=('spectrum','bands')).astype(np.float16)


        ################################
        # original - uncomment to restore
        #################################

        # error_array = LUT - stackedT.astype(np.float16)  # subtract reflectance from snicar reflectance pixelwise
        
        ####################
        # NEW: EXPERIMENTAL
        ####################
        def get_error_array(LUT,ds):
            error_array = LUT-stackedT.astype(np.float32)
            return error_array
        
        error_array = xr.apply_ufunc(get_error_array,LUT,stackedT)

        ###################
        # END
        ###################

        idx_array = xr.apply_ufunc(abs,error_array,dask='allowed')
        idx_array = xr.apply_ufunc(np.mean,idx_array,input_core_dims=[['bands']],dask='allowed',kwargs={'axis':-1})
        idx_array = xr.apply_ufunc(np.argmin,idx_array,input_core_dims=[['spectrum']],dask='allowed',kwargs={'axis':-1})

        # unravel index computes the index in the original n-dimeniona LUT from the index in the flattened LUT
        @dask.delayed
        def unravel(idx_array,densities,grain_rds,algae):
            param_array = np.array(np.unravel_index(idx_array,[len(densities),len(grain_rds),len(algae)]))
            return param_array

        param_array = unravel(idx_array,densities,grain_rds,algae)
        param_array = np.array(param_array.compute(scheduler='processes'))

        # flush disk
        idx_array = None
        error_array = None
        dirtyLUT = None

        #use the indexes to retrieve the actual parameter values for each pixel from the LUT indexes
        # only the first value from the parameter arrays is needed.

        counter = 0
        param_names = ['density','reff','algae']
        
        for param in [densities, grain_rds, algae]:

            for i in np.arange(0,len(param),1):

                if i ==0: # in first loop, pixels !=i should be replaced by param_array values, as result_array doesn't exist yet

                    result_array = np.where(param_array[counter]==i, param[i], param_array[counter])

                else:

                    result_array = np.where(param_array[counter]==i, param[i], result_array)

            # reshape to original dims, add metadata, convert to xr DataArray, apply mask
            result_array = result_array.reshape(int(np.sqrt(len(stackedT))),int(np.sqrt(len(stackedT))))
            resultxr = xr.DataArray(data=result_array,dims=['y','x'], coords={'x':S2vals.x, 'y':S2vals.y}).chunk(2000,2000)
            
            resultxr = resultxr.where(mask2 > 0)

            # send to netcdf and flush memory
            resultxr.to_netcdf(str(os.environ['PROCESS_DIR']+ "{}.nc".format(param_names[counter])))
            
            result_array = None
            resultxr = None
            counter +=1

        
        # retrieved params are saved as temporary netcdfs to the process_dir and then collated directly from 
        # file into the final dataset in run_classifier.py

        return

    def run_ebmodel(self, alb):

           ## Input Data, as per first row of Brock and Arnold (2000) spreadsheet
            lat = 67.0666
            lon = -49.38
            lon_ref = 0
            summertime = 0
            slope = 1.
            aspect = 90.
            elevation = 1020.
            albedo = alb
            roughness = 0.005
            met_elevation = 1020.
            lapse = 0.0065

            day = 202
            time = 1200
            inswrad = 571
            avp = 900
            airtemp = 5.612
            windspd = 3.531

            SWR,LWR,SHF,LHF = ebm.calculate_seb(lat, lon, lon_ref, day, time, summertime, slope, aspect, elevation,
                                                met_elevation, lapse, inswrad, avp, airtemp, windspd, albedo, roughness)

            sw_melt, lw_melt, shf_melt, lhf_melt, total = ebm.calculate_melt(
                SWR,LWR,SHF,LHF, windspd, airtemp)

            # flush memory
            sw_melt = None
            lw_melt = None
            shf_melt = None
            lhf_melt = None
            SWR = None
            LWR = None
            SHF = None
            LHF = None

            return total



