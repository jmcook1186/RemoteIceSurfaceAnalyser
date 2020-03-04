"""
#########################
SNICAR_Inverter.py
Joseph Cook, October 2019
#########################

This script adpts the SNICAR_inverter.py from the BioSNICAR_PY package for integration into BIC.
This is a standalone script used for development and testing, and is not part of the main package.

This is computationally expensive to run over an entire S2 pixel - an Azure DSVM has been provisioned to run
this distributed over 64 cores. On that VM it takes 53 minutes to process a single tile.

for testing, reduce the image size to 250000 pixels or less - this runs ok on JC laptop.
i.e. stackedT = stackedT[0:250000]



TODO: APPLY THE RETRIEVAL TO THE IMAGE *AFTER* THE MASK IS APPLIED TO LIMIT NO OF COMPUTATIONS AND REDUCE RUNTIME

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import dask
import os

##############################################################################
# READ S2 IMAGE
#####################################


def invert_snicar(S2xr):


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

        # reformat LUT: flatten LUT from 3D to 2D array with one column per combination of RT params, one row per wavelength
        LUT = np.load(str(os.environ['PROCESS_DIR'] + 'SNICAR_LUT_2058.npy')).reshape(2058,len(wavelengths))

        # find most similar LUT spectrum for each pixel in S2 image
        # astype(float 16) to reduce memory allocation (default was float64)
        LUT = LUT[:,idx] # reduce wavelengths to only the 9 that match the S2 image

        LUT = xr.DataArray(LUT,dims=('spectrum','bands')).astype(np.float16)

        error_array = LUT - stackedT.astype(np.float16)  # subtract reflectance from snicar reflectance pixelwise
        idx_array = xr.apply_ufunc(abs,error_array,dask='allowed')
        idx_array = xr.apply_ufunc(np.mean,idx_array,input_core_dims=[['bands']],dask='allowed',kwargs={'axis':-1})
        idx_array = xr.apply_ufunc(np.argmin,idx_array,input_core_dims=[['spectrum']],dask='allowed',kwargs={'axis':-1})


        # unravel index computes the index in the original n-dimeniona LUT from the index in the flattened LUT
        @dask.delayed
        def unravel(idx_array,side_lengths,densities,dust,algae):
            param_array = np.array(np.unravel_index(idx_array,[len(side_lengths),len(densities),len(dust),len(algae)]))
            return param_array

        param_array = unravel(idx_array,side_lengths,densities,dust,algae)
        param_array = np.array(param_array.compute())

        print("param_array shape = ",param_array.shape)
        print("param_array type = ",type(param_array))
        
        # flush disk
        idx_array = None
        error_array = None
        dirtyLUT = None

        #use the indexes to retrieve the actual parameter values for each pixel from the LUT indexes
        # since the values are assumed equal in all vertical layers (side_lengths and density) or only the top layer (LAPs)
        # only the first value from the parameter arrays is needed.

        counter = 0
        param_names = ['side_lengths','densities','dust','algae']
        for param in [side_lengths, densities, dust, algae]:

            for i in np.arange(0,len(param),1):

                if i ==0: # in first loop, pixels !=i should be replaced by param_array values, as result_array doesn't exist yet
                    print("result_array gets values {}".format(param[i][0]))
                    result_array = np.where(param_array[counter]==i, param[i][0], param_array[counter])

                else:
                    print("result_array gets values {}".format(param[i][0]))
                    result_array = np.where(param_array[counter]==i, param[i][0], result_array)

            # reshape to original dims, add metadata, convert to xr DataArray, apply mask
            result_array = result_array.reshape(int(np.sqrt(len(stackedT))),int(np.sqrt(len(stackedT))))
            resultxr = xr.DataArray(data=result_array,dims=['y','x'], coords={'x':S2vals.x, 'y':S2vals.y}).chunk(2000,2000)
            
            print("PROCESSED RESULT ARRAY: \n")
            print(resultxr.values)

            # PREVENT ALGAL/DUST OVERESTIMATE IN WATER/CC/SN PIXELS ##
            resultxr = resultxr.where(mask2>0)

            print(result_array)

            # send to netcdf and flush memory
            resultxr.to_netcdf(str(os.environ['PROCESS_DIR']+ "{}.nc".format(param_names[counter])))
            
            result_array = None
            resultxr = None
            counter +=1

        
        # retrieved params are saved as temporary netcdfs to the process_dir and then collated directly from 
        # file into the final dataset in run_classifier.py
    return


def format_retrieved_params(S2xr):

    side_length = xr.load_dataarray(str(os.environ['PROCESS_DIR'] + 'side_lengths.nc'))
    density = xr.load_dataarray(str(os.environ['PROCESS_DIR'] + 'densities.nc'))
    dust = xr.load_dataarray(str(os.environ['PROCESS_DIR'] + 'dust.nc'))
    algae = xr.load_dataarray(str(os.environ['PROCESS_DIR'] + 'algae.nc'))

    
    dataset = xr.Dataset({
        'side_length': (['x', 'y'], side_length.values),
        'density': (['x', 'y'], density.values),
        'dust': (['x', 'y'], dust.values),
        'algae': (['x', 'y'], algae.values)},
        coords={'x': S2xr.x, 'y': S2xr.y})

    return dataset


with xr.open_dataset("/home/joe/Code/BioSNICAR_GO_PY/S2vals.nc",chunks={'x':2000,'y':2000}) as S2xr:

    invert_snicar(S2xr)
    dataset = format_retrieved_params(S2xr)
