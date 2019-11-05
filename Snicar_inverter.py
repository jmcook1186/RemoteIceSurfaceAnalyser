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

##############################################################################
# READ S2 IMAGE
#####################################


def invert_snicar(S2xr):

    band_idx = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9], name='bands')

    # concatenate the bands into a single dimension ('bands_idx') in the data array
    concat = xr.concat([S2xr.B02, S2xr.B03, S2xr.B04, S2xr.B05, S2xr.B06, S2xr.B07,
                        S2xr.B08, S2xr.B11, S2xr.B12], band_idx)

    # stack the values into a 1D array
    stacked = concat.stack(allpoints=['y', 'x'])

    # Transpose and rename so that DataArray has exactly the same layout/labels as the training DataArray.
    # mask out nan areas not masked out by GIMP
    stackedT = stacked.T
    stackedT = stackedT.rename({'allpoints': 'samples'})

    # copy range of values used to generate LUT. THESE MUST MATCH THOSE USED TO BUILD LUT!!!
    side_lengths = [[3000,3000,3000,3000,3000],[5000,5000,5000,5000,5000],[7000,7000,7000,7000,7000],[9000,9000,9000,9000,9000],[12000,12000,12000,12000,12000],[15000,15000,15000,15000,15000]]
    densities = [[300,300,300,300,300],[400,400,400,400,400],[500,500,500,500,500],[600,600,600,600,600],[700,700,700,700,700],[800,800,800,800,800],[900,900,900,900,900]]
    dust = [[10000,0,0,0,0],[50000,0,0,0,0],[100000,0,0,0,0],[500000,0,0,0,0],[1000000,0,0,0,0],[1500000,0,0,0,0],[2000000,0,0,0,0]]
    algae = [[10000,0,0,0,0],[50000,0,0,0,0],[100000,0,0,0,0],[500000,0,0,0,0],[1000000,0,0,0,0],[1500000,0,0,0,0],[2000000,0,0,0,0]]
    wavelengths = np.arange(0.3,5,0.01)

    # set index in wavelength array corresponding to centre WL for each S2 band
    # e.g. for idx[0] == 19, wavelengths[19] = 490 nm
    idx = [19, 26, 36, 40, 44, 48, 56, 131, 190]

    # RECHUNK STACKEDT: choice of chunk size is crucial for maximising speed while preventing memory over-allocation.
    # While 20000 seems small there are very large intermediate arrays spawned by the compute() function.
    # The compute() function takes almost a day to run on an 8 core i7-7000 GHz processor with 32GB RAM. 
    # Takes 58 mins on 64 core Azure VM.

    stackedT = stackedT.chunk(20000,9)
    stackedT = stackedT[1000000:1250000]

    # reformat LUT: flatten to 2D array with column per combination of RT params, row per wavelength
    dirtyLUT = np.load(str(os.environ['PROCESS_DIR'] + 'SNICAR_LUT_2058.npy')).reshape(len(side_lengths)*len(densities)*len(dust)*len(algae),len(wavelengths))
    array = []

    # find most similar SNICAR spectrum for each pixel
    # astype(float 16) to reduce memory allocation (default was float64)
    dirtyLUT = dirtyLUT[:,idx] # reduce wavelengths to match sample spectrum
    dirtyLUT = xr.DataArray(dirtyLUT,dims=('spectrum','bands')).astype(np.float16)
    error_array = dirtyLUT-stackedT.astype(np.float16) # subtract reflectance from snicar reflectance pixelwise

    # average error over bands, then provide index of minimum error (index refers to position in flattened LUT for closest matching spectrum)
    idx_array = xr.apply_ufunc(abs,error_array,dask='allowed').mean(dim='bands').argmin(dim='spectrum') 
    idx_array.compute() # compute delayed product from transformation in previous line

    # unravel index computes the index in the original n-dimeniona LUT from the index in the flattened LUT 
    param_array = np.array(np.unravel_index(idx_array,[len(side_lengths),len(densities),len(dust),len(algae)]))

    # flush disk
    idx_array = None
    error_array = None
    dirtyLUT = None

    #use the indexes to retrieve the actual parameter values for each pixel from the LUT indexes
    # since the values are equal for all layers (side_lengths and density) or only the top layer (LAPs)
    # only the first value from the parameeter arrays are taken.

    counter = 0
    param_names = ['side_lengths','densities','dust','algae']
    for param in [side_lengths, densities, dust, algae]:
        for i in np.arange(0,len(param),1):
            if i ==0:
                result_array = np.where(param_array[counter]==i, param[i][0], param_array[counter])
            else:
                result_array = np.where(param_array[counter]==i, param[i][0], result_array)

        result_array = result_array.reshape([int(np.sqrt(len(stackedT))),int(np.sqrt(len(stackedT)))])
        resultxr = xr.DataArray(result_array)
        result_array = None
        resultxr.to_netcdf(str(os.environ['PROCESS_DIR']+ f"{param_names[counter]}.nc"))
        resultxr = None
        counter +=1

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
