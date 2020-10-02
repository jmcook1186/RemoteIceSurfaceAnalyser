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
plt.style.use('tableau-colorblind10')


#############################################################
#############################################################


def JJA_maps(path, var, year, vmin, vmax, dpi=300):

    wea = xr.open_dataset(str(path+'REDUCED_{}_22wea_{}.nc'.format(var,year)))
    web = xr.open_dataset(str(path+'REDUCED_{}_22web_{}.nc'.format(var,year)))
    wec = xr.open_dataset(str(path+'REDUCED_{}_22wec_{}.nc'.format(var,year)))
    wet = xr.open_dataset(str(path+'REDUCED_{}_22wet_{}.nc'.format(var,year)))
    weu = xr.open_dataset(str(path+'REDUCED_{}_22weu_{}.nc'.format(var,year)))
    wev = xr.open_dataset(str(path+'REDUCED_{}_22wev_{}.nc'.format(var,year)))

    JunStart = str(year+'0601')
    JunEnd = str(year+'0630')
    JulStart = str(year+'0701')
    JulEnd = str(year+'0731')
    AugStart = str(year+'0801')
    AugEnd = str(year+'0831')
    
    wea_jun = wea[var].loc[JunStart:JunEnd].mean(dim='date')
    web_jun = web[var].loc[JunStart:JunEnd].mean(dim='date')
    wec_jun = wec[var].loc[JunStart:JunEnd].mean(dim='date')
    wet_jun = wet[var].loc[JunStart:JunEnd].mean(dim='date')
    weu_jun = weu[var].loc[JunStart:JunEnd].mean(dim='date')
    wev_jun = wev[var].loc[JunStart:JunEnd].mean(dim='date')

    wea_jul = wea[var].loc[JulStart:JulEnd].mean(dim='date')
    web_jul = web[var].loc[JulStart:JulEnd].mean(dim='date')
    wec_jul = wec[var].loc[JulStart:JulEnd].mean(dim='date')
    wet_jul = wet[var].loc[JulStart:JulEnd].mean(dim='date')
    weu_jul = weu[var].loc[JulStart:JulEnd].mean(dim='date')
    wev_jul = wev[var].loc[JulStart:JulEnd].mean(dim='date')

    wea_aug = wea[var].loc[AugStart:AugEnd].mean(dim='date')
    web_aug = web[var].loc[AugStart:AugEnd].mean(dim='date')
    wec_aug = wec[var].loc[AugStart:AugEnd].mean(dim='date')
    wet_aug = wet[var].loc[AugStart:AugEnd].mean(dim='date')
    weu_aug = weu[var].loc[AugStart:AugEnd].mean(dim='date')
    wev_aug = wev[var].loc[AugStart:AugEnd].mean(dim='date')

    plt.close()
    fig,axes = plt.subplots(6,3)
    plt.subplots_adjust(wspace=0.0001,hspace=0.001)
    
    axes[0,0].imshow(wec_jun,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[0,0].set_xticks([],[])
    axes[0,0].set_yticks([],[])

    axes[0,1].imshow(wec_jul,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[0,1].set_xticks([],[])
    axes[0,1].set_yticks([],[])

    axes[0,2].imshow(wec_aug,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[0,2].set_xticks([],[])
    axes[0,2].set_yticks([],[])

    axes[1,0].imshow(web_jun,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[1,0].set_xticks([],[])
    axes[1,0].set_yticks([],[])

    axes[1,1].imshow(web_jul,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[1,1].set_xticks([],[])
    axes[1,1].set_yticks([],[])

    axes[1,2].imshow(web_aug,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[1,2].set_xticks([],[])
    axes[1,2].set_yticks([],[])


    axes[2,0].imshow(wea_jun,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[2,0].set_xticks([],[])
    axes[2,0].set_yticks([],[])

    axes[2,1].imshow(wea_jul,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[2,1].set_xticks([],[])
    axes[2,1].set_yticks([],[])

    axes[2,2].imshow(wea_aug,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[2,2].set_xticks([],[])
    axes[2,2].set_yticks([],[])

    axes[3,0].imshow(wev_jun,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[3,0].set_xticks([],[])
    axes[3,0].set_yticks([],[])

    axes[3,1].imshow(wev_jul, vmin=vmin,vmax=vmax, cmap=cmap)
    axes[3,1].set_xticks([],[])
    axes[3,1].set_yticks([],[])

    axes[3,2].imshow(wev_aug,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[3,2].set_xticks([],[])
    axes[3,2].set_yticks([],[])

    axes[4,0].imshow(weu_jun,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[4,0].set_xticks([],[])
    axes[4,0].set_yticks([],[])

    axes[4,1].imshow(weu_jul,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[4,1].set_xticks([],[])
    axes[4,1].set_yticks([],[])

    axes[4,2].imshow(weu_aug,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[4,2].set_xticks([],[])
    axes[4,2].set_yticks([],[])

    axes[5,0].imshow(wet_jun,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[5,0].set_xticks([],[])
    axes[5,0].set_yticks([],[])

    axes[5,1].imshow(wet_jul,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[5,1].set_xticks([],[])
    axes[5,1].set_yticks([],[])

    axes[5,2].imshow(wet_aug,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[5,2].set_xticks([],[])
    axes[5,2].set_yticks([],[])

    plt.savefig(str(path+'/JJA_{}_{}.jpg'.format(var,year)), dpi = dpi)
    
    return


def JJA_stats(path, var, year):

    wea = xr.open_dataset(str(path+'REDUCED_{}_22wea_{}.nc'.format(var,year)))
    web = xr.open_dataset(str(path+'REDUCED_{}_22web_{}.nc'.format(var,year)))
    wec = xr.open_dataset(str(path+'REDUCED_{}_22wec_{}.nc'.format(var,year)))
    wet = xr.open_dataset(str(path+'REDUCED_{}_22wet_{}.nc'.format(var,year)))
    weu = xr.open_dataset(str(path+'REDUCED_{}_22weu_{}.nc'.format(var,year)))
    wev = xr.open_dataset(str(path+'REDUCED_{}_22wev_{}.nc'.format(var,year)))

    JunStart = str(year+'0601')
    JunEnd = str(year+'0630')
    JulStart = str(year+'0701')
    JulEnd = str(year+'0731')
    AugStart = str(year+'0801')
    AugEnd = str(year+'0831')


    for month in ['Jun','Jul','Aug']:

        values = []
        count = []

        if month == 'Jun':
            Start = JunStart
            End = JunEnd
        elif month == 'Jul':
            Start = JulStart
            End = JulEnd
        elif month == 'Aug':
            Start = AugStart
            End = AugEnd

        # CALC MEAN

        for tile in [wev]:
            
            tile = tile[var].loc[Start:End].mean(dim='date')
            tile_sum = tile.sum()
            tile_count = tile.count()
            values.append(tile_sum.values)
            count.append(tile_count.values)

        val = np.array(values).sum()
        co = np.array(count).sum()

        dz_mean = val/co
        print(dz_mean)


        # CALC STDEV

        SD_list = []

        for tile in [wev]:
            
            tile = tile[var].loc[Start:End].mean(dim='date')
            tile = (tile - dz_mean)**2

            SD_list.append(tile.sum())

       
        tile_sum = (np.array(SD_list).sum()) / co
        SD = np.sqrt(tile_sum)

        print("STDEV = ", SD)
            
    return


def annual_stats(path, var, year):
    
    wea = xr.open_dataset(str(path+'REDUCED_{}_22wea_{}.nc'.format(var,year)))
    web = xr.open_dataset(str(path+'REDUCED_{}_22web_{}.nc'.format(var,year)))
    wec = xr.open_dataset(str(path+'REDUCED_{}_22wec_{}.nc'.format(var,year)))
    wet = xr.open_dataset(str(path+'REDUCED_{}_22wet_{}.nc'.format(var,year)))
    weu = xr.open_dataset(str(path+'REDUCED_{}_22weu_{}.nc'.format(var,year)))
    wev = xr.open_dataset(str(path+'REDUCED_{}_22wev_{}.nc'.format(var,year)))

    values = []
    count = []
    maxlist = []
    minlist = []

    for tile in [wea, web, wec, wet, weu, wev]:
        
        tile = tile[var].mean(dim='date')
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
        
        tile = tile[var].mean(dim='date')
        tile = (tile - dz_mean)**2

        SD_list.append(tile.sum())


    tile_sum = (np.array(SD_list).sum()) / co
    SD = np.sqrt(tile_sum)

    print("STDEV = ", SD)
            
    return


def annual_maps(path, var, year, vmin, vmax, dpi=300):
    
    wea = xr.open_dataset(str(path+'REDUCED_{}_22wea_{}.nc'.format(var,year)))
    web = xr.open_dataset(str(path+'REDUCED_{}_22web_{}.nc'.format(var,year)))
    wec = xr.open_dataset(str(path+'REDUCED_{}_22wec_{}.nc'.format(var,year)))
    wet = xr.open_dataset(str(path+'REDUCED_{}_22wet_{}.nc'.format(var,year)))
    weu = xr.open_dataset(str(path+'REDUCED_{}_22weu_{}.nc'.format(var,year)))
    wev = xr.open_dataset(str(path+'REDUCED_{}_22wev_{}.nc'.format(var,year)))

    wea_mean = wea[var].mean(dim='date')
    web_mean = web[var].mean(dim='date')
    wec_mean = wec[var].mean(dim='date')
    wet_mean = wet[var].mean(dim='date')
    weu_mean = weu[var].mean(dim='date')
    wev_mean = wev[var].mean(dim='date')

    fig,axes = plt.subplots(6,1)
    plt.subplots_adjust(wspace=0.000001,hspace=0.001)
    
    axes[0].imshow(wec_mean,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[0].set_xticks([],[])
    axes[0].set_yticks([],[])

    axes[1].imshow(web_mean,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[1].set_xticks([],[])
    axes[1].set_yticks([],[])

    axes[2].imshow(wea_mean,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[2].set_xticks([],[])
    axes[2].set_yticks([],[])

    axes[3].imshow(wev_mean,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[3].set_xticks([],[])
    axes[3].set_yticks([],[])

    axes[4].imshow(weu_mean,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[4].set_xticks([],[])
    axes[4].set_yticks([],[])

    axes[5].imshow(wet_mean,vmin=vmin,vmax=vmax, cmap=cmap)
    axes[5].set_xticks([],[])
    axes[5].set_yticks([],[])

    plt.savefig(str(path+'annual_mean_{}.jpg'.format(year)),dpi=dpi)

    return

def annual_histograms(path, var, year):

    wea = xr.open_dataset(str(path+'REDUCED_{}_22wea_{}.nc'.format(var,year)))
    web = xr.open_dataset(str(path+'REDUCED_{}_22web_{}.nc'.format(var,year)))
    wec = xr.open_dataset(str(path+'REDUCED_{}_22wec_{}.nc'.format(var,year)))
    wet = xr.open_dataset(str(path+'REDUCED_{}_22wet_{}.nc'.format(var,year)))
    weu = xr.open_dataset(str(path+'REDUCED_{}_22weu_{}.nc'.format(var,year)))
    wev = xr.open_dataset(str(path+'REDUCED_{}_22wev_{}.nc'.format(var,year)))

    wea_mean = wea[var].mean(dim='date')
    web_mean = web[var].mean(dim='date')
    wec_mean = wec[var].mean(dim='date')
    wet_mean = wet[var].mean(dim='date')
    weu_mean = weu[var].mean(dim='date')
    wev_mean = wev[var].mean(dim='date')

    tot = xr.merge([wea_mean,web_mean,wec_mean,wet_mean,weu_mean,wev_mean])
    tot = tot/0.94 *(1/0.917)/10

    plt.hist(np.ravel(tot.to_array()),bins=100)
    plt.xlim(0, 30000)
    plt.ylim(0, 2.5E7)
    plt.ylabel('Frequency')
    plt.xlabel('Algae concentration (cells/mL)')
    plt.savefig(str(path+'histogram_{}.jpg'.format(year)),dpi=dpi)

    return



def plot_BandRatios(savepath):

    BandRatios = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/BandRatios.csv')

    DBA2 = BandRatios[BandRatios['Index']=='2DBA']
    DBA3 = BandRatios[BandRatios['Index']=='3DBA']
    NDCI = BandRatios[BandRatios['Index']=='NCDI']
    MCI = BandRatios[BandRatios['Index']=='MCI']
    II = BandRatios[BandRatios['Index']=='II']

    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize = (5,15))
    
    # plot each curve individually in first panel to enable
    # assigning labels to curves for legend
    ax1.plot(DBA2['Grain'], DBA2.loc[:,'0ppb'], marker='x',label = '0')
    ax1.plot(DBA2['Grain'], DBA2.loc[:,'10000ppb'], marker='x',label = '10000 ppb')
    ax1.plot(DBA2['Grain'], DBA2.loc[:,'20000ppb'], marker='x',label = '20000 ppb')
    ax1.plot(DBA2['Grain'], DBA2.loc[:,'30000ppb'], marker='x',label = '30000 ppb')
    ax1.plot(DBA2['Grain'], DBA2.loc[:,'40000ppb'], marker='x',label = '40000 ppb')
    ax1.plot(DBA2['Grain'], DBA2.loc[:,'50000ppb'], marker='x',label = '50000 ppb')
    ax1.legend(ncol=3,bbox_to_anchor=(1,1.3))

    ax1.set_xticks(DBA2['Grain'])
    ax1.set_xticklabels(DBA2['Grain'])
    ax1.set_ylabel('Index Value: 2DBA')

    ax2.plot(DBA2['Grain'], DBA3.loc[:,'0ppb':'50000ppb'], marker='x')
    ax2.set_xticks(DBA3['Grain'])
    ax2.set_xticklabels(DBA3['Grain'])
    ax2.set_ylabel('Index Value: 3DBA')

    ax3.plot(DBA2['Grain'], NDCI.loc[:,'0ppb':'50000ppb'], marker='x')
    ax3.set_xticks(NDCI['Grain'])
    ax3.set_xticklabels(NDCI['Grain'])
    ax3.set_ylabel('Index Value: NCDI')

    ax4.plot(DBA2['Grain'], MCI.loc[:,'0ppb':'50000ppb'], marker='x')
    ax4.set_xticks(MCI['Grain'])
    ax4.set_xticklabels(MCI['Grain'])
    ax4.set_ylabel('Index Value: MCI')

    ax5.plot(DBA2['Grain'], II.loc[:,'0ppb':'50000ppb'], marker='x')
    ax5.set_xticks(II['Grain'])
    ax5.set_xticklabels(II['Grain'])
    ax5.set_ylabel('Index Value: II')
    ax5.set_xlabel('Grain size (microns)')

    fig.tight_layout()
    
    plt.savefig(str(savepath+'/BandRatios.png'),dpi=300)

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


def time_series(path, var):

    for year in ['2016','2017','2018','2019']:

        ds = xr.open_dataset('/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/REDUCED_{}_22wev_{}.nc'.format(var,year))
        ds2 = xr.open_dataset('/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/REDUCED_{}_22wev_{}.nc'.format('classified',year))
        
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

        df.area1 = np.array(area1means)/0.94 *(1/0.917)/10
        df2.area1SN = (np.array(area1SN) * 0.0004 / (200*200*0.0004))*100
        df.area2 = np.array(area2means)/0.94 *(1/0.917)/10
        df2.area2SN = (np.array(area2SN) * 0.0004 / (200*200*0.0004))*100
        df.area3 = np.array(area3means)/0.94 *(1/0.917)/10
        df2.area3SN = (np.array(area3SN) * 0.0004 / (200*200*0.0004))*100
        df.area4 = np.array(area4means)/0.94 *(1/0.917)/10
        df2.area4SN = (np.array(area4SN) * 0.0004 / (200*200*0.0004))*100
        df.area5 = np.array(area5means)/0.94 *(1/0.917)/10
        df2.area5SN = (np.array(area5SN) * 0.0004 / (200*200*0.0004))*100
        df.area6 = np.array(area5means)/0.94 *(1/0.917)/10
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


def colorbar(vmin, vmax, cmap):
    """
    function to create a jpg colorbar for building annual and
    JJA map figures

    """

    vmin = vmin/0.94 *(1/0.917)/10
    vmax = vmax/0.94 *(1/0.917)/10

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
year = '2019'
var='algae'
path = str('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/')
dpi = 300
vmin = 0
vmax = 200000
cmap = 'viridis'


# FUNCTION CALLS (uncomment as needed)

#JJA_maps(path, var, year, vmin, vmax, dpi=dpi)
#annual_maps(path, var, year, vmin, vmax, dpi=300)
#annual_stats(path, var, year)
#plot_BandRatios(savepath)
#JJA_stats(path, var, year)
#time_series(path, var = 'algae')
#colorbar(vmin.vmax,cmap)