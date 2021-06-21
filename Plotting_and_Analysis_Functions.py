"""

After main BISC code has been run and the output directory populated with .nc datasets, this script
can be run to plot the desired outputs. The prerequisite is .nc files in the process_dir/output/ directory.

NOTE: best not to run this script in VScode as it tend to persist the "dark theme" of the editor into 
the saved figures, over-riding kwargs in cals to pyplot. Best to run this in Pycharm or from terminal.

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
import re
import pandas as pd
import dask
from scipy import stats

plt.style.use('tableau-colorblind10')


#############################################################
#############################################################
# 2016 J, 2016Ju, 2016 Au
# 2017 J, 2017Ju, 2017 Au
# 2018 J, 2018Ju, 2018 Au
# 2019 J, 2019Ju, 2019 Au

def create_monthly_mean_datasets(year, tile, savepath):
    """

    creates individual datasets for each month, averaging over all dates in month

    """

    path = '/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/'.format(tile.lower())
    ds = xr.open_dataset(str(path+'{}/FULL_OUTPUT_{}_{}_Final.nc'.format(tile.lower(),tile.lower(),year)))

    ds.dz.values[ds.dz.values==1]=0.1 # correct error where in one LUT 0.1 was typed 1.0

    JunDates = [date for date in ds.date.values if f"{year}06" in date]
    JulDates = [date for date in ds.date.values if f"{year}07" in date]
    AugDates = [date for date in ds.date.values if f"{year}08" in date]

    if len(JunDates) >0:
        dsJun = ds.sel(date=JunDates).mean(dim='date')
        dsJun.to_netcdf(savepath+"/{}/{}_June_{}.nc".format(tile.lower(),tile,year))
    else:
        print("No valid data for June ({},{})".format(tile,year))
    
    if len(JulDates) > 0:
        dsJul = ds.sel(date=JulDates).mean(dim='date')
        dsJul.to_netcdf(savepath+"{}/{}_July_{}.nc".format(tile.lower(),tile,year))
    else:
        print("No valid data for July ({},{})".format(tile,year))
    
    if len(AugDates) > 0:
        dsAug = ds.sel(date=AugDates).mean(dim='date')
        dsAug.to_netcdf(savepath+"{}/{}_Aug_{}.nc".format(tile.lower(),tile,year))
    else:
        print("No valid data for August ({},{})".format(tile,year))
    
    return


def create_annual_mean_datasets(year, tile, savepath):

    path = '/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/'.format(tile.lower())
    ds = xr.open_dataset(str(path+'{}/FULL_OUTPUT_{}_{}_Final.nc'.format(tile.lower(),tile.lower(),year)))
    ds.dz.values[ds.dz.values==1]=0.1

    # do it piecemeal to avoid 31 x 5490 x 5490 array being loaded into memory
    ds_alg = ds.algae.mean(dim='date',skipna=True)
    ds_reff = ds.reff.mean(dim='date',skipna=True)
    ds_density = ds.density.mean(dim='date',skipna=True)
    ds_albedo = ds.albedo.mean(dim='date',skipna=True)
    ds_dz = ds.dz.mean(dim='date',skipna=True)
    ds_mask = ds.FinalMask.sel(date=ds.date[0])

    # grab lat/lon from original ds
    lat = ds.latitude.sel(date=ds.date[0])
    lon = ds.longitude.sel(date=ds.date[0])

    # collate means into new ds
    dsMean = xr.Dataset({
        'albedo': (['x', 'y'], ds_albedo),
        'density': (['x','y'], ds_density),
        'dz':(['x','y'],ds_dz),
        'reff':(['x','y'], ds_reff),
        'algae':(['x','y'],ds_alg),
        'mask':(['x','y'],ds_mask),
        'longitude': (['x', 'y'], lon),
        'latitude': (['x', 'y'], lat),
    },
        coords={'x': ds.x, 'y': ds.y})

    # save to nc
    print("now saving dataset")
    dsMean.to_netcdf(savepath+"{}/{}_MEAN_{}.nc".format(tile.lower(),tile,year))
    
    return

def annual_maps(path, var, year, vmin, vmax, cmap = 'BuPu', dpi=300):

    wea = xr.open_dataset(str(path+'/22wea/22WEA_MEAN_{}.nc'.format(year)))
    web = xr.open_dataset(str(path+'/22web/22WEB_MEAN_{}.nc'.format(year)))
    wec = xr.open_dataset(str(path+'/22wec/22WEC_MEAN_{}.nc'.format(year)))
    wet = xr.open_dataset(str(path+'/22wet/22WET_MEAN_{}.nc'.format(year)))
    weu = xr.open_dataset(str(path+'/22weu/22WEU_MEAN_{}.nc'.format(year)))
    wev = xr.open_dataset(str(path+'/22wev/22WEV_MEAN_{}.nc'.format(year)))


    fig,axes = plt.subplots(6,1)
    plt.subplots_adjust(wspace=0.000001,hspace=0.001)
    axes[0].imshow(wec[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[0].set_xticks([],[])
    axes[0].set_yticks([],[])

    axes[1].imshow(web[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[1].set_xticks([],[])
    axes[1].set_yticks([],[])

    axes[2].imshow(wea[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[2].set_xticks([],[])
    axes[2].set_yticks([],[])

    axes[3].imshow(wev[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[3].set_xticks([],[])
    axes[3].set_yticks([],[])

    axes[4].imshow(weu[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[4].set_xticks([],[])
    axes[4].set_yticks([],[])

    axes[5].imshow(wet[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[5].set_xticks([],[])
    axes[5].set_yticks([],[])
    
    plt.savefig(str(path+'annual_mean{}_{}.jpg'.format(var,year)),dpi=dpi)

    return


def create_monthly_maps(path, var, year, vmin, vmax, cmap ='BuPu', dpi=300):

    wea_jun = xr.open_dataset(str(path+'/22wea/22WEA_June_{}.nc'.format(year)))
    web_jun = xr.open_dataset(str(path+'/22web/22WEB_June_{}.nc'.format(year)))
    wec_jun = xr.open_dataset(str(path+'/22wec/22WEC_June_{}.nc'.format(year)))
    wet_jun = xr.open_dataset(str(path+'/22wet/22WET_June_{}.nc'.format(year)))
    weu_jun = xr.open_dataset(str(path+'/22weu/22WEU_June_{}.nc'.format(year)))
    wev_jun = xr.open_dataset(str(path+'/22wev/22WEV_June_{}.nc'.format(year)))

    wea_jul = xr.open_dataset(str(path+'/22wea/22WEA_July_{}.nc'.format(year)))
    web_jul = xr.open_dataset(str(path+'/22web/22WEB_July_{}.nc'.format(year)))
    wec_jul = xr.open_dataset(str(path+'/22wec/22WEC_July_{}.nc'.format(year)))
    wet_jul = xr.open_dataset(str(path+'/22wet/22WET_July_{}.nc'.format(year)))
    weu_jul = xr.open_dataset(str(path+'/22weu/22WEU_July_{}.nc'.format(year)))
    wev_jul = xr.open_dataset(str(path+'/22wev/22WEV_July_{}.nc'.format(year)))

    wea_aug = xr.open_dataset(str(path+'/22wea/22WEA_Aug_{}.nc'.format(year)))
    web_aug = xr.open_dataset(str(path+'/22web/22WEB_Aug_{}.nc'.format(year)))
    if (year != '2016') & (year!= '2018'):
        wec_aug = xr.open_dataset(str(path+'/22wec/22WEC_Aug_{}.nc'.format(year)))
    wet_aug = xr.open_dataset(str(path+'/22wet/22WET_Aug_{}.nc'.format(year)))
    weu_aug = xr.open_dataset(str(path+'/22weu/22WEU_Aug_{}.nc'.format(year)))
    wev_aug = xr.open_dataset(str(path+'/22wev/22WEV_Aug_{}.nc'.format(year)))

    
    fig,axes = plt.subplots(6,3)
    plt.subplots_adjust(wspace=0.0001,hspace=0.001)
    
    axes[0,0].imshow(wec_jun[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[0,0].set_xticks([],[])
    axes[0,0].set_yticks([],[])

    axes[0,1].imshow(wec_jul[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[0,1].set_xticks([],[])
    axes[0,1].set_yticks([],[])

    if (year != '2016') & (year!= '2018'):
        axes[0,2].imshow(wec_aug[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[0,2].set_xticks([],[])
    axes[0,2].set_yticks([],[])

    axes[1,0].imshow(web_jun[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[1,0].set_xticks([],[])
    axes[1,0].set_yticks([],[])

    axes[1,1].imshow(web_jul[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[1,1].set_xticks([],[])
    axes[1,1].set_yticks([],[])

    axes[1,2].imshow(web_aug[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[1,2].set_xticks([],[])
    axes[1,2].set_yticks([],[])


    axes[2,0].imshow(wea_jun[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[2,0].set_xticks([],[])
    axes[2,0].set_yticks([],[])

    axes[2,1].imshow(wea_jul[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[2,1].set_xticks([],[])
    axes[2,1].set_yticks([],[])

    axes[2,2].imshow(wea_aug[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[2,2].set_xticks([],[])
    axes[2,2].set_yticks([],[])

    axes[3,0].imshow(wev_jun[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[3,0].set_xticks([],[])
    axes[3,0].set_yticks([],[])

    axes[3,1].imshow(wev_jul[var], vmin=vmin,vmax=vmax, cmap=cmap)
    axes[3,1].set_xticks([],[])
    axes[3,1].set_yticks([],[])

    axes[3,2].imshow(wev_aug[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[3,2].set_xticks([],[])
    axes[3,2].set_yticks([],[])

    axes[4,0].imshow(weu_jun[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[4,0].set_xticks([],[])
    axes[4,0].set_yticks([],[])

    axes[4,1].imshow(weu_jul[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[4,1].set_xticks([],[])
    axes[4,1].set_yticks([],[])

    axes[4,2].imshow(weu_aug[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[4,2].set_xticks([],[])
    axes[4,2].set_yticks([],[])

    axes[5,0].imshow(wet_jun[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[5,0].set_xticks([],[])
    axes[5,0].set_yticks([],[])

    axes[5,1].imshow(wet_jul[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[5,1].set_xticks([],[])
    axes[5,1].set_yticks([],[])

    axes[5,2].imshow(wet_aug[var],vmin=vmin,vmax=vmax, cmap=cmap)
    axes[5,2].set_xticks([],[])
    axes[5,2].set_yticks([],[])

    plt.savefig(str(path+'/JJA_{}_{}.jpg'.format(var,year)), dpi = dpi)
    
    return



def JJA_stats(path, var, year):

    wea_jun = xr.open_dataset(str(path+'/22wea/22WEA_June_{}.nc'.format(year)))[var]
    web_jun = xr.open_dataset(str(path+'/22web/22WEB_June_{}.nc'.format(year)))[var]
    wec_jun = xr.open_dataset(str(path+'/22wec/22WEC_June_{}.nc'.format(year)))[var]
    wet_jun = xr.open_dataset(str(path+'/22wet/22WET_June_{}.nc'.format(year)))[var]
    weu_jun = xr.open_dataset(str(path+'/22weu/22WEU_June_{}.nc'.format(year)))[var]
    wev_jun = xr.open_dataset(str(path+'/22wev/22WEV_June_{}.nc'.format(year)))[var]

    wea_jul = xr.open_dataset(str(path+'/22wea/22WEA_July_{}.nc'.format(year)))[var]
    web_jul = xr.open_dataset(str(path+'/22web/22WEB_July_{}.nc'.format(year)))[var]
    wec_jul = xr.open_dataset(str(path+'/22wec/22WEC_July_{}.nc'.format(year)))[var]
    wet_jul = xr.open_dataset(str(path+'/22wet/22WET_July_{}.nc'.format(year)))[var]
    weu_jul = xr.open_dataset(str(path+'/22weu/22WEU_July_{}.nc'.format(year)))[var]
    wev_jul = xr.open_dataset(str(path+'/22wev/22WEV_July_{}.nc'.format(year)))[var]

    wea_aug = xr.open_dataset(str(path+'/22wea/22WEA_Aug_{}.nc'.format(year)))[var]
    web_aug = xr.open_dataset(str(path+'/22web/22WEB_Aug_{}.nc'.format(year)))[var]
    wet_aug = xr.open_dataset(str(path+'/22wet/22WET_Aug_{}.nc'.format(year)))[var]
    weu_aug = xr.open_dataset(str(path+'/22weu/22WEU_Aug_{}.nc'.format(year)))[var]
    wev_aug = xr.open_dataset(str(path+'/22wev/22WEV_Aug_{}.nc'.format(year)))[var]

    # wec aug data missing for 2016 and 2018 due to cloud cover, only include in 2017 2019
    if (year != '2016') & (year!= '2018'):    
        wec_aug = xr.open_dataset(str(path+'/22wec/22WEC_Aug_{}.nc'.format(year)))[var]

    jun_sum = wea_jun.sum() + web_jun.sum() + wec_jun.sum() + wet_jun.sum() + weu_jun.sum() + wev_jun.sum()
    jul_sum = wea_jul.sum() + web_jul.sum() + wec_jul.sum() + wet_jul.sum() + weu_jul.sum() + wev_jul.sum()
    aug_sum = wea_aug.sum() + web_aug.sum() + wet_aug.sum() + weu_aug.sum() + wev_aug.sum()

    if (year != '2016') & (year!= '2018'):
        aug_sum = aug_sum + wec_aug.sum()
    
    jun_count = wea_jun.count() + web_jun.count() + wec_jun.count() + wet_jun.count() + weu_jun.count() + wev_jun.count()
    jul_count = wea_jul.count() + web_jul.count() + wec_jul.count() + wet_jul.count() + weu_jul.count() + wev_jul.count()
    aug_count = wea_aug.count() + web_aug.count() + wet_aug.count() + weu_aug.count() + wev_aug.count()

    if (year != '2016') & (year!= '2018'):
        aug_count = aug_count + wec_aug.count()

    jun_mean = jun_sum/jun_count
    jul_mean = jul_sum/jul_count
    aug_mean = aug_sum/aug_count


    jun_std = np.mean([wea_jun.std(), web_jun.std(), wec_jun.std(), wet_jun.std(), weu_jun.std(), wev_jun.std()]) 
    jul_std = np.mean([wea_jul.std(), web_jul.std(), wec_jul.std(), wet_jul.std(), weu_jul.std(), wev_jul.std()])
    
    if (year != '2016') & (year!= '2018'):
        aug_std = np.mean([wea_aug.std(), web_aug.std(), wec_aug.std(), wet_aug.std(), weu_aug.std(), wev_aug.std()])
    else:
        aug_std = np.mean([wea_aug.std(), web_aug.std(), wet_aug.std(), weu_aug.std(), wev_aug.std()])

    print("jun mean = ", jun_mean.values)
    print("jun STDEV = ", jun_std)
            
    print("jul mean = ", jul_mean.values)
    print("jul STDEV = ", jul_std)

    print("aug mean = ", aug_mean.values)
    print("aug STDEV = ", aug_std)

    return


def correlations_btwn_vars(var1, var2, year, path):
    
    wea1 = xr.DataArray(xr.open_dataset(str(path+'/22wea/22WEA_MEAN_{}.nc'.format(year)))[var1])
    web1 = xr.DataArray(xr.open_dataset(str(path+'/22web/22WEB_MEAN_{}.nc'.format(year)))[var1])
    wec1 = xr.DataArray(xr.open_dataset(str(path+'/22wec/22WEC_MEAN_{}.nc'.format(year)))[var1])
    wet1 = xr.DataArray(xr.open_dataset(str(path+'/22wet/22WET_MEAN_{}.nc'.format(year)))[var1])
    weu1 = xr.DataArray(xr.open_dataset(str(path+'/22weu/22WEU_MEAN_{}.nc'.format(year)))[var1])
    wev1 = xr.DataArray(xr.open_dataset(str(path+'/22wev/22WEV_MEAN_{}.nc'.format(year)))[var1])

    wea2 = xr.DataArray(xr.open_dataset(str(path+'/22wea/22WEA_MEAN_{}.nc'.format(year)))[var2])
    web2 = xr.DataArray(xr.open_dataset(str(path+'/22web/22WEB_MEAN_{}.nc'.format(year)))[var2])
    wec2 = xr.DataArray(xr.open_dataset(str(path+'/22wec/22WEC_MEAN_{}.nc'.format(year)))[var2])
    wet2 = xr.DataArray(xr.open_dataset(str(path+'/22wet/22WET_MEAN_{}.nc'.format(year)))[var2])
    weu2 = xr.DataArray(xr.open_dataset(str(path+'/22weu/22WEU_MEAN_{}.nc'.format(year)))[var2])
    wev2 = xr.DataArray(xr.open_dataset(str(path+'/22wev/22WEV_MEAN_{}.nc'.format(year)))[var2])

    def format_data(ds1, ds2):

        A = ds1.where((ds1.values>0)&(ds2.values>0))
        B = ds2.where((ds1.values>0)&(ds2.values>0))
        A = np.ravel(A.values)
        B = np.ravel(B.values)
        A = A[~np.isnan(A)]
        B = B[~np.isnan(B)]

        return A, B

    wea1, wea2 = format_data(wea1,wea2)
    web1, web2 = format_data(web1,web2)
    wec1, wec2 = format_data(wec1, wec2)
    wet1, wet2 = format_data(wet1, wet2)
    weu1, weu2 = format_data(weu1, weu2)
    wev1, wev2 = format_data(wev1, wev2)

    weaR, weaP = stats.pearsonr(wea1,wea2)
    webR, webP = stats.pearsonr(web1,web2)
    wecR, wecP = stats.pearsonr(wec1,wec2)
    wetR, wetP = stats.pearsonr(wet1,wet2)
    weuR, weuP = stats.pearsonr(weu1,weu2)
    wevR, wevP = stats.pearsonr(wev1,wev2)
    
    R = np.mean([weaR,webR,wecR,wetR,weuR,wevR])
    p = np.mean([weaP,webP,wecP,wetP,weuP,wevP])

    print("WEA: ", weaR,weaP)
    print("WEB: ", webR,webP)
    print("WEC: ", wecR,wecP)
    print("WET: ", wetR,wetP)
    print("WEU: ", weuR,weuP)
    print("WEV: ", wevR,wevP)
    print("DZ MEAN: ", R,p)

    return

def corr_heatmaps(path):

    import seaborn  as sns
    import matplotlib.pyplot as plt

    # ['algae', 'albedo', 'dz', 'density']
    
    d2016 = [[1, -0.57, -0.76, 0.144, 0.32],
    [-0.57, 1, 0.88, -0.71, -0.786],
    [-0.76, 0.88, 1, -0.47, -0.61],
    [0.144, -0.71, -0.47, 1, 0.83],
    [0.32, -0.786, -0.61, 0.83, 1]]

    d2017 =[[1, -0.35, -0.60, 0.11, 0.2],
    [-0.35, 1, 0.87, -0.71, -0.714],
    [-0.60,0.87,1,-0.58, -0.67],
    [0.11, -0.71, -0.58, 1, 0.86],
    [0.2, -0.714, -0.67, 0.86, 1]]

    d2018 = [[1, -0.22, 0.53, 0.06, 0.14],
    [-0.22, 1, 0.84, -0.71, -0.77],
    [0.53, 0.84, 1, -0.58, -0.68],
    [0.06, -0.71, -0.58, 1, 0.87],
    [0.14, -0.77, -0.68, 0.87,1]]

    d2019 = [[1, -0.56, -0.73, 0.14, 0.3],
    [-0.56, 1, 0.87, -0.75, -0.84],
    [-0.73, 0.87, 1, -0.48, -0.62],
    [0.14, -0.75, -0.48, 1, 0.84],
    [0.3, -0.84, -0.62, 0.84,1]]

    f,(ax1,ax2,ax3,ax4,axcb) = plt.subplots(1,5,gridspec_kw={'width_ratios':[1,1,1,1,0.08]},figsize=(15,5))
    
    # for a in [ax1, ax2, ax3, ax4]:
    #     a.set_aspect(1)

    ax1.get_shared_y_axes().join(ax2,ax3, ax4)
    g1 = sns.heatmap(d2016,cmap="vlag",annot=True,cbar=False,ax=ax1,vmin=-1, vmax=1)
    g1.set_ylabel('')
    g1.set_xlabel('')
    ax1.set_title("2016")

    g2 = sns.heatmap(d2017,cmap="vlag",annot=True,cbar=False,ax=ax2,vmin=-1, vmax=1)
    g2.set_ylabel('')
    g2.set_xlabel('')
    ax2.set_title("2017")


    g3 = sns.heatmap(d2018,cmap="vlag",annot=True,cbar=False,ax=ax3,vmin=-1, vmax=1)
    g3.set_ylabel('')
    g3.set_xlabel('')
    ax3.set_title("2018")

    g4 = sns.heatmap(d2019,cmap="vlag",annot=True,cbar_ax=axcb,ax=ax4,vmin=-1, vmax=1)
    g4.set_ylabel('')
    g4.set_xlabel('')
    ax4.set_title("2019")

    axcb.set_ylabel("Pearson's R")

    for ax in [g1,g2,g3,g4]:
        tl = ['algae', 'albedo', 'dz', 'density','r_eff']
        ax.set_xticklabels(tl, rotation=90)
        ax.set_yticklabels([])
        ax.set_yticks([])

    g1.set_yticks([0,1,2,3,4])
    g1.set_yticklabels(tl, rotation=45)

    plt.tight_layout()

    plt.savefig(str(path+'/PearsonsR_heatmap.jpg'))
    
    return


def annual_stats(path, var, year):
    
    wea = xr.open_dataset(str(path+'/22wea/22WEA_MEAN_{}.nc'.format(year)))[var]
    web = xr.open_dataset(str(path+'/22web/22WEB_MEAN_{}.nc'.format(year)))[var]
    wec = xr.open_dataset(str(path+'/22wec/22WEC_MEAN_{}.nc'.format(year)))[var]
    wet = xr.open_dataset(str(path+'/22wet/22WET_MEAN_{}.nc'.format(year)))[var]
    weu = xr.open_dataset(str(path+'/22weu/22WEU_MEAN_{}.nc'.format(year)))[var]
    wev = xr.open_dataset(str(path+'/22wev/22WEV_MEAN_{}.nc'.format(year)))[var]

    values = []
    count = []
    maxlist = []
    minlist = []

    for tile in [wea, web, wec, wet, weu, wev]:
        
        tile_sum = tile.sum()
        tile_count = tile.count()
        values.append(tile_sum.values)
        count.append(tile_count.values)
        maxlist.append(tile.max().values)
        minlist.append(tile.min().values)

    val = np.array(values).sum()
    co = np.array(count).sum()

    dz_mean = val/co
    print("mean = ",dz_mean)
    print("min = ", np.array(minlist).min())
    print("max = ", np.array(maxlist).max())

    # CALC STDEV

    SD_list = []

    for tile in [wea, web, wec, wet, weu, wev]:
        
        SD_list.append(tile.std())


    SD = np.mean(SD_list)

    print("STDEV = ", SD)
            
    return


def annual_histograms(path, var, year, dpi = 300):

    
    wea = xr.open_dataset(str(path+'/22wea/22WEA_MEAN_{}.nc'.format(year)))[var]
    web = xr.open_dataset(str(path+'/22web/22WEB_MEAN_{}.nc'.format(year)))[var]
    wec = xr.open_dataset(str(path+'/22wec/22WEC_MEAN_{}.nc'.format(year)))[var]
    wet = xr.open_dataset(str(path+'/22wet/22WET_MEAN_{}.nc'.format(year)))[var]
    weu = xr.open_dataset(str(path+'/22weu/22WEU_MEAN_{}.nc'.format(year)))[var]
    wev = xr.open_dataset(str(path+'/22wev/22WEV_MEAN_{}.nc'.format(year)))[var]

    tot = xr.merge([wea,web,wec,wet,weu,wev]).to_array()
    tot = np.ravel(tot.values)
    tot=tot[~np.isnan(tot)]
    tot = tot[tot>0]
    tot = tot/ (np.pi*(4**2*40)*0.0014*0.3*(1/0.917))

    plt.hist(tot,bins=30)
    plt.xlim(0, 30000)
    plt.ylim(0, 2.5E7)
    plt.ylabel('Frequency')
    plt.xlabel('Algae concentration (cells/mL)')
    plt.savefig(str(path+'histogram_{}.jpg'.format(year)),dpi=dpi)
    plt.close()

    return


def time_series(path, var):
    for year in ['2016','2017','2018','2019']:

        ds = xr.open_dataset('/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/22wev/FULL_OUTPUT_22wev_{}_Final.nc'.format(year))
        ds2 = xr.open_dataset('/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/22wev/FULL_OUTPUT_22wev_{}_Final.nc'.format(year))
        
        area1 = ds[dict(y=slice(4200, 4400),x=slice(1500,1700))]
        area2 = ds[dict(y=slice(4000, 4200),x=slice(2000,2200))]
        area3 = ds[dict(y=slice(4000, 4200),x=slice(2300,2500))]
        area4 = ds[dict(y=slice(4300, 4500),x=slice(3500,3700))]
        area5 = ds[dict(y=slice(4300, 4500),x=slice(4500,4700))]
        area6 = ds[dict(y=slice(4700, 4900),x=slice(4500,4700))]
        

        area1class = ds2[dict(y=slice(4200, 4400),x=slice(1500,1700))]
        area2class = ds2[dict(y=slice(4000, 4200),x=slice(2000,2200))]
        area3class = ds2[dict(y=slice(4000, 4200),x=slice(2300,2500))]
        area4class = ds2[dict(y=slice(4300, 4500),x=slice(3500,3700))]
        area5class = ds2[dict(y=slice(4300, 4500),x=slice(4500,4700))]
        area6class = ds2[dict(y=slice(4700, 4900),x=slice(4500,4700))]
        
        area1means = []
        area1SN = []
        area2means = []
        area2SN = []
        area3means = []
        area3SN = []
        area4means = []
        area4SN = []
        area5means = []
        area5SN = []
        area6means = []
        area6SN = []

        for i in ds.date:

            area1means.append(area1[var].loc[i.values].mean().values)
            area2means.append(area2[var].loc[i.values].mean().values)
            area3means.append(area3[var].loc[i.values].mean().values)
            area4means.append(area4[var].loc[i.values].mean().values)
            area5means.append(area5[var].loc[i.values].mean().values)            
            area6means.append(area6[var].loc[i.values].mean().values)
            
        
        for ii in ds2.date:
            area1SN.append(area1class.classified.loc[ii.values].where(area1class.classified.loc[ii.values].values==1).count().values)
            area2SN.append(area1class.classified.loc[ii.values].where(area2class.classified.loc[ii.values].values==1).count().values)
            area3SN.append(area1class.classified.loc[ii.values].where(area3class.classified.loc[ii.values].values==1).count().values)
            area4SN.append(area1class.classified.loc[ii.values].where(area4class.classified.loc[ii.values].values==1).count().values)
            area5SN.append(area5class.classified.loc[ii.values].where(area5class.classified.loc[ii.values].values==1).count().values)
            area6SN.append(area6class.classified.loc[ii.values].where(area6class.classified.loc[ii.values].values==1).count().values)

                
        df = pd.DataFrame(columns=['date','area1','area2','area3','area4','area5','area6'])
        df.date = ds.date.values      
        df.date = pd.to_datetime(df.date)

        df2 = pd.DataFrame(columns=['date','area1SN','area2SN','area3SN','area4SN','area5SN','area6SN'])
        df2.date = ds2.date.values      
        df2.date = pd.to_datetime(df2.date)

        df.area1 = np.array(area1means)/ (np.pi*(4**2*40)*0.0014*0.3*(1/0.917))
        df2.area1SN = (np.array(area1SN) * 0.0004 / (200*200*0.0004))*100
        df.area2 = np.array(area2means)/ (np.pi*(4**2*40)*0.0014*0.3*(1/0.917))
        df2.area2SN = (np.array(area2SN) * 0.0004 / (200*200*0.0004))*100
        df.area3 = np.array(area3means)/ (np.pi*(4**2*40)*0.0014*0.3*(1/0.917))
        df2.area3SN = (np.array(area3SN) * 0.0004 / (200*200*0.0004))*100
        df.area4 = np.array(area4means)/ (np.pi*(4**2*40)*0.0014*0.3*(1/0.917))
        df2.area4SN = (np.array(area4SN) * 0.0004 / (200*200*0.0004))*100
        df.area5 = np.array(area5means)/ (np.pi*(4**2*40)*0.0014*0.3*(1/0.917))
        df2.area5SN = (np.array(area5SN) * 0.0004 / (200*200*0.0004))*100
        df.area6 = np.array(area5means)/ (np.pi*(4**2*40)*0.0014*0.3*(1/0.917))
        df2.area6SN = (np.array(area6SN) * 0.0004 / (200*200*0.0004))*100

        r = pd.date_range(start=df.date.min(), end=df.date.max())
        df.set_index('date').reindex(r).rename_axis('date').reset_index(inplace=True)
        df.to_csv(str(path+'/DF_{}.csv'.format(year)),index=None)

        r2 = pd.date_range(start=df2.date.min(), end=df2.date.max())
        df2.set_index('date').reindex(r2).rename_axis('date').reset_index(inplace=True)
        df2.to_csv(str(path+'/DF_{}_CLASS.csv'.format(year)),index=None)


    DF2016 = pd.read_csv(str(path+'/DF_2016.csv'))
    DF2016.date = pd.to_datetime(DF2016.date)
    DF2016 = DF2016[DF2016["date"].isin(pd.date_range("2016-06-01", "2016-08-18"))]

    DF2016Class = pd.read_csv(str(path+'/DF_2016_CLASS.csv'))
    DF2016Class.date = pd.to_datetime(DF2016Class.date)
    DF2016Class = DF2016Class[DF2016Class["date"].isin(pd.date_range("2016-06-01", "2016-08-18"))]

    DF2017 = pd.read_csv(str(path+'DF_2017.csv'))
    DF2017.date = pd.to_datetime(DF2017.date)
    DF2017 = DF2017[DF2017["date"].isin(pd.date_range("2017-06-01", "2017-08-18"))]

    DF2017Class = pd.read_csv(str(path+'/DF_2017_CLASS.csv'))
    DF2017Class.date = pd.to_datetime(DF2017Class.date)
    DF2017Class = DF2017Class[DF2017Class["date"].isin(pd.date_range("2017-06-01", "2017-08-18"))]

    DF2018 = pd.read_csv(str(path+'/DF_2018.csv'))
    DF2018.date = pd.to_datetime(DF2018.date)
    DF2018 = DF2018[DF2018["date"].isin(pd.date_range("2018-06-01", "2018-08-18"))]

    DF2018Class = pd.read_csv(str(path+'/DF_2018_CLASS.csv'))
    DF2018Class.date = pd.to_datetime(DF2018Class.date)
    DF2018Class = DF2018Class[DF2018Class["date"].isin(pd.date_range("2018-06-01", "2018-08-18"))]

    DF2019 = pd.read_csv(str(path+'/DF_2019.csv'))
    DF2019.date = pd.to_datetime(DF2019.date)
    DF2019 = DF2019[DF2019["date"].isin(pd.date_range("2019-06-01", "2019-08-18"))]

    DF2019Class = pd.read_csv(str(path+'/DF_2019_CLASS.csv'))
    DF2019Class.date = pd.to_datetime(DF2019Class.date)
    DF2019Class = DF2019Class[DF2019Class["date"].isin(pd.date_range("2019-06-01", "2019-08-18"))]
    
    
    plt.close()
    fig, axes = plt.subplots(2,2,figsize=(12,10))

    axes[0,0].plot(DF2016.date,DF2016.area1,label='area1')
    axes[0,0].plot(DF2016.date,DF2016.area2,label='area2')
    axes[0,0].plot(DF2016.date,DF2016.area3,label='area3')
    axes[0,0].plot(DF2016.date,DF2016.area4,label='area4')
    axes[0,0].plot(DF2016.date,DF2016.area5,label='area5')
    axes[0,0].plot(DF2016.date,DF2016.area6,label='area6')
    axes[0,0].legend(loc='upper left',ncol=3)

    ax0 = axes[0,0].twinx()
    ax0.plot(DF2016Class.date,DF2016Class.area1SN, linestyle='None', marker = 'x')
    ax0.plot(DF2016Class.date,DF2016Class.area2SN, linestyle='None', marker = 'x')
    ax0.plot(DF2016Class.date,DF2016Class.area3SN, linestyle='None', marker = 'x')
    ax0.plot(DF2016Class.date,DF2016Class.area4SN, linestyle='None', marker = 'x')
    ax0.plot(DF2016Class.date,DF2016Class.area5SN, linestyle='None', marker = 'x')
    ax0.plot(DF2016Class.date,DF2016Class.area6SN, linestyle='None', marker = 'x')

    axes[0,1].plot(DF2017.date,DF2017.area1)
    axes[0,1].plot(DF2017.date,DF2017.area2)
    axes[0,1].plot(DF2017.date,DF2017.area3)
    axes[0,1].plot(DF2017.date,DF2017.area4)
    axes[0,1].plot(DF2017.date,DF2017.area5)
    axes[0,1].plot(DF2017.date,DF2017.area6)

    ax1 = axes[0,1].twinx()
    ax1.plot(DF2017Class.date,DF2017Class.area1SN, linestyle='None', marker = 'x')
    ax1.plot(DF2017Class.date,DF2017Class.area2SN, linestyle='None', marker = 'x')
    ax1.plot(DF2017Class.date,DF2017Class.area3SN, linestyle='None', marker = 'x')
    ax1.plot(DF2017Class.date,DF2017Class.area4SN, linestyle='None', marker = 'x')
    ax1.plot(DF2017Class.date,DF2017Class.area5SN, linestyle='None', marker = 'x')
    ax1.plot(DF2017Class.date,DF2017Class.area6SN, linestyle='None', marker = 'x')

    axes[1,0].plot(DF2018.date,DF2018.area1)
    axes[1,0].plot(DF2018.date,DF2018.area2)
    axes[1,0].plot(DF2018.date,DF2018.area3)
    axes[1,0].plot(DF2018.date,DF2018.area4)
    axes[1,0].plot(DF2018.date,DF2018.area5)
    axes[1,0].plot(DF2018.date,DF2018.area6)

    ax2 = axes[1,0].twinx()
    ax2.plot(DF2018.date,DF2018Class.area1SN, linestyle='None', marker = 'x')
    ax2.plot(DF2018.date,DF2018Class.area2SN, linestyle='None', marker = 'x')
    ax2.plot(DF2018.date,DF2018Class.area3SN, linestyle='None', marker = 'x')
    ax2.plot(DF2018.date,DF2018Class.area4SN, linestyle='None', marker = 'x')
    ax2.plot(DF2018.date,DF2018Class.area5SN, linestyle='None', marker = 'x')
    ax2.plot(DF2018.date,DF2018Class.area6SN, linestyle='None', marker = 'x')

    axes[1,1].plot(DF2019.date,DF2019.area1)
    axes[1,1].plot(DF2019.date,DF2019.area2)
    axes[1,1].plot(DF2019.date,DF2019.area3)
    axes[1,1].plot(DF2019.date,DF2019.area4)
    axes[1,1].plot(DF2019.date,DF2019.area5)
    axes[1,1].plot(DF2019.date,DF2019.area6)

    ax3 = axes[1,1].twinx()
    ax3.plot(DF2019Class.date,DF2019Class.area1SN, linestyle='None', marker = 'x')
    ax3.plot(DF2019Class.date,DF2019Class.area2SN, linestyle='None', marker = 'x')
    ax3.plot(DF2019Class.date,DF2019Class.area3SN, linestyle='None', marker = 'x')
    ax3.plot(DF2019Class.date,DF2019Class.area4SN, linestyle='None', marker = 'x')
    ax3.plot(DF2019Class.date,DF2019Class.area5SN, linestyle='None', marker = 'x')
    ax3.plot(DF2019Class.date,DF2019Class.area6SN, linestyle='None', marker = 'x')

    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%d%m%y')

    axes[0,0].set_xticklabels(DF2016.date, rotation=60)
    axes[0,0].xaxis.set_major_formatter(myFmt)
    axes[0,0].set_ylim(0,25000)
    
    axes[0,1].set_xticklabels(DF2017.date, rotation=60)
    axes[0,1].xaxis.set_major_formatter(myFmt)
    axes[0,1].set_ylim(0,25000)

    axes[1,0].set_xticklabels(DF2018.date, rotation=60)
    axes[1,0].xaxis.set_major_formatter(myFmt)
    axes[1,0].set_ylim(0,25000)

    axes[1,1].set_xticklabels(DF2019.date, rotation=60)
    axes[1,1].xaxis.set_major_formatter(myFmt)
    axes[1,1].set_ylim(0,25000)

    fig.tight_layout()

    plt.savefig(str(path+'/time_series.jpg'),dpi=300)
    return










def add_boxes_to_rgb():

    import matplotlib.image as img
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    im = img.imread('/home/joe/Code/QGIS/merged.tif')

    fig, ax = plt.subplots(1)

    rect1 = patches.Rectangle((4200,1500),200,200, edgecolor='k', facecolor="none")
    rect2 = patches.Rectangle((4000,2300),200,200, edgecolor='k', facecolor="none")
    rect3 = patches.Rectangle((4300,3500),200,200, edgecolor='k', facecolor="none")
    rect4 = patches.Rectangle((4300,4500),200,200, edgecolor='k', facecolor="none")
    rect5 = patches.Rectangle((4700,4500),200,200, edgecolor='k', facecolor="none")
    ax.add_patch(rect1),ax.add_patch(rect2),ax.add_patch(rect3),ax.add_patch(rect4),ax.add_patch(rect5)

    ax.imshow(im)

    plt.savefig('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/rgb_boxes.jpg',dpi=300)

    return



def colorbar(vmin, vmax, cmap):
    """
    function to create a jpg colorbar for building annual and
    JJA map figures

    """

    vmin = vmin / (np.pi*(4**2*40)*0.0014*0.3*(1/0.917)*10)
    vmax = vmax / (np.pi*(4**2*40)*0.0014*0.3*(1/0.917)*10)

    fig, ax = plt.subplots(1, 1)

    fraction = 1

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, pad=.05, extend='both', fraction=fraction)

    ax.axis('off')
    
    plt.savefig(str(path+'colorbar.jpg'),dpi=300)
    
    return


# USER DEFINED VARIABLES
# year = '2016'
# var='albedo'
# path = str('/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/')
# dpi = 150
# vmin = 0
# vmax = 1
# cmap = 'viridis'


# # FUNCTION CALLS (uncomment as needed)

# JJA_maps(path, var, year, vmin, vmax, dpi=dpi)
#annual_maps(path, var, year, vmin, vmax, dpi=300)
#annual_stats(path, var, year)
#plot_BandRatios(savepath)
#JJA_stats(path, var, year)
#time_series(path, var = 'algae')

# for year in ['2016','2017','2018','2019']:
#     annual_histograms(path, var, year)
# colorbar(vmin,vmax,cmap)