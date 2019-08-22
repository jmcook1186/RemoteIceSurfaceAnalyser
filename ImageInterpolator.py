
"""
VERSION 0.1- **first pass** at synthetic image generator

script uses a pixel-wise linear interpolation to infill missing images (i.e. create a completely synthetic image for dates where
real images are unavailable due to excessive cloud cover or insufficient ice cover or a download problem. This is
achieved by identifying the missing dates, then identifying the closest past and closest future "good" images.

For each pixel in the 5490 x 5490 images, 2 values are available - a past and future value. These are regressed against
the DOY (i.e. a straight line is fit between past and future). The regression equation is then used to predict the
value for the same pixel for the missing DOY.

"""

# TODO: decide what to do about the classification layer!
    # current ideas = a) a "nostalgic classifier" where if past != future, the earlier class is preferred, or b) an
    # "impatient classifier" where the future class is prioritised, c) devise some albedo-based metric to determine
    # threshold for updating class
# TODO: refactor into function and update paths etc so that image interpolator can be called from main driver script
# TODO: work out how to deal with pixels that are NaN in past and have a value in future and vice-versa
# TODO: combine with intra-image interpolator (i.e. infilling cloudy or missing pixels)
    # idea: apply interpolation (e.g. bicubic) to all NaNs before masking out non-ice areas
# TODO: save the synthetic data as an xarray dataset with appropriate metadata to match those for the real dates

import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime

plt.ioff()

# identify tile and date for target image
tile = "22wev"
date = "20160703"
seasonStart = "2016_06_01"
seasonEnd = "2016_06_30"

# get list of nc files and extract the dates as strings
DateList = glob.glob("/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/{}*.nc".format(tile))
fmt = '%Y_%m_%d' # set format for date string
DOYlist = [] # empty list to append days of year

# create list of all DOYs between start and end of season (1st June to 31st Aug as default)
DOYstart = datetime.datetime.strptime(seasonStart,fmt).timetuple().tm_yday
DOYend = datetime.datetime.strptime(seasonEnd,fmt).timetuple().tm_yday
FullSeason = np.arange(DOYstart,DOYend,1)

# find dates with images in image repo
# strip out the date from the filename and insert underscores to separate out YYYY, MM, DD
for i in np.arange(0,len(DateList),1):
    DateList[i] = DateList[i][68:-34]
    DateList[i] = str(DateList[i][0:4]+'_'+DateList[i][4:6]+'_'+DateList[i][6:8]) # format string to match format defined above
    DOYlist.append(datetime.datetime.strptime(DateList[i], fmt).timetuple().tm_yday)

# compare full season DOY list with DOY list in image repo
DOYlist = np.array(DOYlist)
MissingDates = np.setdiff1d(FullSeason, DOYlist)
year = seasonStart[0:4]

# for each missing date find the nearest past and future "good" images in the repo
for Missing in MissingDates:

    test = DOYlist - Missing #subtract the missing DOY from each DOY in the image repo
    closestPast = DOYlist[np.where(test<0, test, -9999).argmax()] # find the closest image from the past
    closestFuture = DOYlist[np.where(test>0, test, 9999).argmin()] # find the closest image int he future
    closestPastString = datetime.datetime.strptime(str(year[2:4] + str(closestPast)),
                                                   '%y%j').date().strftime('%Y%m%d') # convert to string format
    closestFutureString = datetime.datetime.strptime(str(year[2:4] + str(closestFuture)),
                                                     '%y%j').date().strftime('%Y%m%d') # convert to string format
    MissingDateString = datetime.datetime.strptime(str(year[2:4] + str(Missing)),
                                                     '%y%j').date().strftime('%Y%m%d') # convert to string format

    print("closestPast = {}, MissingDate ={}, closestFuture = {}".format(closestPastString,MissingDateString,closestFutureString))
    # load past and future images that will be used for interpolation
    imagePast = xr.open_dataset(str('/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/22wev_'+closestPastString+'_Classification_and_Albedo_Data.nc'))
    imageFuture = xr.open_dataset(str('/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/22wev_'+closestFutureString+'_Classification_and_Albedo_Data.nc'))

    # extract the albedo layer.
    arrayPast = imagePast.albedo.values
    arrayFuture = imageFuture.albedo.values

    # linear regression pixelwise
    slopes = (arrayPast - arrayFuture) / (closestFuture - closestPast)
    intercepts = arrayPast-(slopes*closestPast)
    newImage = slopes * closestPast + intercepts

    maskPast = np.isnan(arrayPast)
    maskFuture = np.isnan(arrayFuture)
    combinedMask = np.ma.mask_or(maskPast, maskFuture)

    newImage = np.ma.array(newImage, mask=combinedMask)
    newImage = newImage.filled(np.nan)
    plt.imshow(newImage)
    plt.savefig('/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/22wev_{}_synthetic_test.png'.format(MissingDateString))
    plt.close()


    ### # some lines useful for clearing folders etc during development ###
    # import os
    # list = glob.glob('/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/*synthetic*')
    # for i in list:
    #     os.remove(i)
