
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
seasonStart = "2016_06_01" # first date in measurement period
seasonEnd = "2016_06_30" # last date in measurement period

# get list of "good" nc files and extract the dates as strings
DateList = glob.glob("/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/{}*.nc".format(tile))
fmt = '%Y_%m_%d' # set format for date string
DOYlist = [] # empty list ready to receive days of year

# create list of all DOYs between start and end of season (1st June to 31st Aug as default)
DOYstart = datetime.datetime.strptime(seasonStart,fmt).timetuple().tm_yday # convert seasonStart to numeric DOY
DOYend = datetime.datetime.strptime(seasonEnd,fmt).timetuple().tm_yday # convert seasonEnd to numeric DOY
FullSeason = np.arange(DOYstart,DOYend,1) # create list starting at DOY for seasonStart and ending at DOY for seasonEnd

# Now create list of DOYs and loist of strings for all the dates in the image repo, i.e. dates with "good images"
# strip out the date from the filename and insert underscores to separate out YYYY, MM, DD so formats are consistent
for i in np.arange(0,len(DateList),1):
    DateList[i] = DateList[i][68:-34]
    DateList[i] = str(DateList[i][0:4]+'_'+DateList[i][4:6]+'_'+DateList[i][6:8]) # format string to match format defined above
    DOYlist.append(datetime.datetime.strptime(DateList[i], fmt).timetuple().tm_yday)

# compare full season DOY list with DOY list in image repo to identify only the missing dates between seasonStart and seasonEnd
DOYlist = np.array(DOYlist)
MissingDates = np.setdiff1d(FullSeason, DOYlist) #setdiff1d identifies the dtaes present in Fullseason but not DOYlist
year = seasonStart[0:4] # the year is the first 4 characters in the string and is needed later

for Missing in MissingDates:
    # for each missing date find the nearest past and future "good" images in the repo
    test = DOYlist - Missing # subtract the missing DOY from each DOY in the image repo
    closestPast = DOYlist[np.where(test<0, test, -9999).argmax()] # find the closest image from the past
    closestFuture = DOYlist[np.where(test>0, test, 9999).argmin()] # find the closest image in the future
    closestPastString = datetime.datetime.strptime(str(year[2:4] + str(closestPast)),
                                                   '%y%j').date().strftime('%Y%m%d') # convert to string format
    closestFutureString = datetime.datetime.strptime(str(year[2:4] + str(closestFuture)),
                                                     '%y%j').date().strftime('%Y%m%d') # convert to string format
    MissingDateString = datetime.datetime.strptime(str(year[2:4] + str(Missing)),
                                                     '%y%j').date().strftime('%Y%m%d') # convert to string format
    # report past, missing and future dates to console
    print("closestPast = {}, MissingDate ={}, closestFuture = {}".format(closestPastString,MissingDateString,closestFutureString))

    # load past and future images that will be used for interpolation
    imagePast = xr.open_dataset(str('/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/22wev_'+closestPastString+'_Classification_and_Albedo_Data.nc'))
    imageFuture = xr.open_dataset(str('/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/22wev_'+closestFutureString+'_Classification_and_Albedo_Data.nc'))

    # extract the albedo layer.
    arrayPast = imagePast.albedo.values
    arrayFuture = imageFuture.albedo.values

    # linear regression pixelwise (2D array of slopes, 2D array of intercepts, independent variable = DOY, solve for pixel value)
    slopes = (arrayPast - arrayFuture) / (closestFuture - closestPast)
    intercepts = arrayPast-(slopes*closestPast)
    newImage = slopes * Missing + intercepts

    # generate mask that eliminates pixels that are NaNs in EITHER past or future image
    maskPast = np.isnan(arrayPast)
    maskFuture = np.isnan(arrayFuture)
    combinedMask = np.ma.mask_or(maskPast, maskFuture)
    combinedMaskNum = combinedMask.astype('uint8')

    # apply mask to synthetic image
    newImage = np.ma.array(newImage, mask=combinedMask)
    newImage = newImage.filled(np.nan)
    plt.imshow(newImage)

    plt.savefig('/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/22wev_{}_synthetic_test.png'.format(MissingDateString))
    plt.close()


    # ### # some lines useful for clearing folders etc during development ###
    # import os
    # list = glob.glob('/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/22wev/*synthetic*')
    # for i in list:
    #     os.remove(i)
