import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

dataset = xr.open_dataset('/home/joe/Desktop/BIC_outputs/22wev_20160701_Classification_and_Albedo_Data.nc')

counter  = 0

layers = [dataset.classified, dataset.albedo]
mask = dataset.FinalMask

for layer in layers:

    x = np.arange(0,layer.shape[1])
    y = np.arange(0,layer.shape[0])

    arr = np.ma.masked_invalid(layer)
    xx,yy = np.meshgrid(x,y)

    x1 = xx[~arr.mask]
    y1 = yy[~arr.mask]
    newlayer = arr[~arr.mask]
    GD1 = interpolate.griddata((x1,y1),newlayer.ravel(),(xx,yy),method='nearest')

#     layer = np.ravel(layer.values)
#     indexes = np.arange(layer.shape[0])
#     good = np.where(np.isfinite(layer))
#     f = interpolate.interp1d(indexes[good],layer[good],bounds_error=False)
#     layer = np.where(np.isfinite(layer),layer,f(indexes))
#     layer = layer.reshape(5490,5490)
#     np.where(mask !=1,layer,np.nan)

    if counter == 0:
        dataset.classified.values = layer
        layer = None

    elif counter == 1:
        dataset.albedo.values = layer
        layer = None

    elif counter == 2:
        dataset.grain_size.values = layer
        layer = None

    elif counter == 3:
        dataset.density.values = layer
        layer = None

    elif counter == 4:
        dataset.dust.values = layer
        layer = None
    
    elif counter == 5:
        dataset.algae.values = layer
        layer = None

    else:
        print ("ERROR IN PIXELWISE CLOUD INTERPOLATION: counter out of range")

    counter +=1
    

dataset = dataset.where(mask ==0)
