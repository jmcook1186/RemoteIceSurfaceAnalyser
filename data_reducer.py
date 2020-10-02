"""

After main BISC code has been run and the output directory populated with .nc datasets, this script
can be run to create smaller individual .nc files for individual variables. This is usefule because
it is much easier to handle smaller <10GB .nc files than the larger ~100GB files that contain all
variables and dates. The plotting scripts in this repository use the reduced files rather than accessing
the data via the large all-variable files.

As well as separating files by variable, this script also removes dates that did not pass manual
quality control.

The prerequisite for this script to run is the "FULL_OUTPUT... .nc" files in the process_dir/output/ directory.

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
import re
import gc

plt.style.use('tableau-colorblind10')


def reduce_outputs(tile, year, dateList, var, savepath):

    ds = xr.open_dataset('/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/{}/FULL_OUTPUT_{}_{}_Final.nc'.format(tile,tile,year))

    ds2 = ds[var].drop_sel(date=dateList)

    ds2.to_netcdf(str(savepath+'REDUCED_' + var + '_' + tile + '_' + year+'.nc'))

    ds = None
    ds2 = None

    gc.collect()

    print("reduced ds {} saved to disk".format(tile))

    return 




for tile in ["22wea","22web","22wec","22wet","22weu","22wev"]:

    for year in ['2016','2017','2018','2019']:

        var = 'Index2DBA'
        savepath = '/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/'

        if year =='2016':
            if tile == "22wea":
                dateList = ['20160613', '20160625', '20160628', '20160701', '20160704', '20160716',\
                            '20160719', '20160809', '20160818', '20160821', '20160824', '20160827']

            elif tile == "22web":
                dateList = ['20160613','20160616','20160710','20160716','20160719','20160728','20160731','20160809',\
                            '20160812','20160824','20160827']
            
            elif tile == "22wec":
                dateList = []

            elif tile == "22wet":
                dateList = ['20160604','20160607','20160610','20160619','20160622','20160625','20160628','20160701','20160704','20160707',\
                            '20160710','20160731','20160803','20160824','20160827','20160830']

            elif tile == "22weu":
                dateList = ['20160601','20160604','20160616','20160628','20160701','20160704','20160707','20160716','20160719','20160722',\
                            '20160818','20160821','20160824','20160827','20160830']

            elif tile == "22wev":
                dateList = ['20160601','20160604','20160616','20160625','20160628','20160701','20160710','20160713','20160716','20160719','20160722','20160809',\
                            '20160812','20160821','20160824','20160827','20160830']

        elif year =='2017':
            if tile == "22wea":
                dateList = ['20170601','20170607','20170610','20170613','20170616','20170619','20170625','20170716','20170719','20170806','20170821','20170824',\
                            '20170827','20170830']

            elif tile == "22web":
                dateList = ['20170604','20170616','20170622','20170625','20170628','20170701','20170707','20170716','20170725','20170809','20170818','20170821',\
                            '20170824','20170827','20170830']
            
            elif tile == "22wec":
                dateList = ['20170601','20170604','20170607','20170610','20170809']

            elif tile == "22wet":
                dateList = ['20170601','20170616','20170619','20170622','20170625','20170704','20170707','20170713','20170716','20170719','20170725','20170803',\
                            '20170809','20170818','20170821']

            elif tile == "22weu":
                dateList = ['20170704','20170707','20170713','20170716','20170722','20170725','20170728','20170815','20170818','20170821','20170824','20170827','20170830']

            elif tile == "22wev":
                dateList = ['20170601','20170722','20170725','20170728','20170731','20170803','20170806','20170824','20170827','20170830']


        elif year =='2018':
            if tile == "22wea":
                dateList = ['20180613','20180704','20180728','20180731']

            elif tile == "22web":
                dateList = ['20180601','20180604','20180607','20180610','20180625','20180719','20180722','20180725','20180806',\
                            '20180809','20180812','20180815','20180818','20180821']
            
            elif tile == "22wec":
                dateList = ['20180601','20180604','20180607','20180610']

            elif tile == "22wet":
                dateList = ['20180601','20180604','20180607','20180610','20180716','20180725']

            elif tile == "22weu":
                dateList = ['20180601','20180604','20180607','20180610','20180613','20180619','20180622','20180625','20180628','20180701','20180704',\
                            '20180707','20180716']

            elif tile == "22wev":
                dateList = ['20180601','20180604','20180607','20180719']



        elif year =='2019':
            if tile == "22wea":
                dateList = ['20190610','20190622','20190625','20190701','20190719','20190827']

            elif tile == "22web":
                dateList = ['20190610','20190616','20190713','20190716','20190719','20190806','20190809','20190821','20190824']
            
            elif tile == "22wec":
                dateList = ['20190722','20190725','20190728','20190815','20190827']

            elif tile == "22wet":
                dateList = ['20190601','20190604','20190607','20190625','20190628','20190701','20190704','20190728','20190731','20190815']

            elif tile == "22weu":
                dateList = ['20190601','20190604','20190607','20190710','20190713','20190722','20190725','20190728','20190731','20190827','20190830']

            elif tile == "22wev":
                dateList = ['20190601','20190604','20190607','20190619','20190716','20190719','20190722']


        
        reduce_outputs(tile,year,dateList,var,savepath)
    
