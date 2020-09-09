import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy 
from scipy import stats # For in-built method to get PCC
from scipy.stats import shapiro
from scipy.stats import chi2_contingency
from scipy.stats import mstats
import statsmodels.formula.api as smf
from matplotlib.dates import DateFormatter



def import_csvs(filepath, DZ = True, Tiles = True, byClass = True, BandRatios = True):
    """

    imports csvs from file and returns pandas dataframes for each S2 tile and
    the total DZ
    
    params:
    
        - DZ: Boolean toggling whether to load in DZ (average values for whole dark zone)
        - Tiles: Boolean toggling whether to load in individual tile data
        - filepath: path to the folder containing the csv files
    
    returns:
        - DZ: pandad dataframe containing variable values for whole DZ
        - Tiles (22wea...22wev): pandas dataframes containing variable values for individual S2 tiles
    """

    if DZ:

        # import dark zone stats
        DZ = pd.read_csv(str(filepath+'WholeDZ.csv'))
        DZ = DZ.sort_values(by='Date').reset_index(drop=True)
        DZ.Date = pd.to_datetime(DZ.Date)

        if byClass:

            DZ_Class = pd.read_csv(str(filepath+'WholeDZ_byClass.csv'))
            DZ_Class = DZ_Class.sort_values(by='Date').reset_index(drop=True)
            DZ_Class.Date = DZ.Date
    else:
         DZ = None
         DZ_Class = None

    if Tiles:

        # import individual tiles
        wea = pd.read_csv(str(filepath+'/22wea.csv'))
        web = pd.read_csv(str(filepath+'/22web.csv'))
        wec = pd.read_csv(str(filepath+'/22wec.csv'))
        wet = pd.read_csv(str(filepath+'/22wet.csv'))
        weu = pd.read_csv(str(filepath+'/22weu.csv'))
        wev = pd.read_csv(str(filepath+'/22wev.csv'))

    else: 
        wea = None
        web =None
        wec = None
        wet = None
        weu = None
        wev = None

    if BandRatios:

        BandRatios = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/BandRatios.csv')

    return DZ, DZ_Class, wea, web, wec, wet, weu, wev, BandRatios



def plot_BandRatios(BandRatios, savepath):

    DBA2 = BandRatios[BandRatios['Index']=='2DBA']
    DBA3 = BandRatios[BandRatios['Index']=='3DBA']
    NDCI = BandRatios[BandRatios['Index']=='NCDI']
    MCI = BandRatios[BandRatios['Index']=='MCI']
    II = BandRatios[BandRatios['Index']=='II']
    DBA2_2 = BandRatios[BandRatios['Index']=='2DBA2']

    fig, axes = plt.subplots(2,3, figsize=(20,10))
    
    # plot each curve individually in first panel to enable
    # assigning labels to curves for legend
    axes[0,0].plot(DBA2['Grain'], DBA2.loc[:,'0ppb'], marker='x',label = '0')
    axes[0,0].plot(DBA2['Grain'], DBA2.loc[:,'10000ppb'], marker='x',label = '10000 ppb')
    axes[0,0].plot(DBA2['Grain'], DBA2.loc[:,'20000ppb'], marker='x',label = '20000 ppb')
    axes[0,0].plot(DBA2['Grain'], DBA2.loc[:,'30000ppb'], marker='x',label = '30000 ppb')
    axes[0,0].plot(DBA2['Grain'], DBA2.loc[:,'40000ppb'], marker='x',label = '40000 ppb')
    axes[0,0].plot(DBA2['Grain'], DBA2.loc[:,'50000ppb'], marker='x',label = '50000 ppb')
    axes[0,0].legend(ncol=3,bbox_to_anchor=(1,1.3))

    axes[0,0].set_xticks(DBA2['Grain'])
    axes[0,0].set_xticklabels(DBA2['Grain'])
    axes[0,0].set_ylabel('Index Value: 2DBA')

    axes[0,1].plot(DBA2['Grain'], DBA3.loc[:,'0ppb':'50000ppb'], marker='x')
    axes[0,1].set_xticks(DBA3['Grain'])
    axes[0,1].set_xticklabels(DBA3['Grain'])
    axes[0,1].set_ylabel('Index Value: 3DBA')

    axes[0,2].plot(DBA2['Grain'], NDCI.loc[:,'0ppb':'50000ppb'], marker='x')
    axes[0,2].set_xticks(NDCI['Grain'])
    axes[0,2].set_xticklabels(NDCI['Grain'])
    axes[0,2].set_ylabel('Index Value: NCDI')

    axes[1,0].plot(DBA2['Grain'], MCI.loc[:,'0ppb':'50000ppb'], marker='x')
    axes[1,0].set_xticks(MCI['Grain'])
    axes[1,0].set_xticklabels(MCI['Grain'])
    axes[1,0].set_ylabel('Index Value: MCI')

    axes[1,1].plot(DBA2['Grain'], II.loc[:,'0ppb':'50000ppb'], marker='x')
    axes[1,1].set_xticks(II['Grain'])
    axes[1,1].set_xticklabels(II['Grain'])
    axes[1,1].set_ylabel('Index Value: II')
    axes[1,1].set_xlabel('Grain size (microns)')

    axes[1,2].plot(DBA2_2['Grain'], DBA2_2.loc[:,'0ppb':'50000ppb'], marker='x')
    axes[1,2].set_xticks(II['Grain'])
    axes[1,2].set_xticklabels(II['Grain'])
    axes[1,2].set_ylabel('Index Value: 2DBA-2')
    axes[1,2].set_xlabel('Grain size (microns)')

    fig.tight_layout()
    
    plt.savefig(str(savepath+'/BandRatios.png'),dpi=300)

    return


def timeseries_STDerrorbars(DZ, wea, web, wec, wet, weu, wev, savepath):

    """
    
    function plots the time series for each variable 
    (albedo, grain size, density, algae...)
    in separate subplots in a 5 panel figure. 
    The values for the total DZ and for each individual 
    tile are plotted on the same axes for comparison.

    params:
        - DZ: pandas dataframe containing time series for each variable 
            across the whole dark zone
        - wea...wev: pandas dataframes containing time series for each 
            variable for each individual tile
        - savepath: path to save figure to
    returns:
        - None

    """
    # 1) How do the variables change over time?
    # solid black line for whole DZ, coloured lines for individual tiles

    fig,axes = plt.subplots(nrows = 5,ncols = 4,figsize = (20,15))
    every_nth = 2 #spacing between x labels

    DZ2016 = DZ[DZ.Date.between('2016-06-01','2016-08-31')]

    DZ2017 = DZ[DZ.Date.between('2017-06-01','2017-08-31')]
    
    DZ2018 = DZ[DZ.Date.between('2018-06-01','2018-08-31')]
    
    DZ2019 = DZ[DZ.Date.between('2019-06-01','2019-08-31')]

    ###################
    # SUBPLOT 1: Albedo

    axes[0,0].errorbar(DZ2016.Date,DZ2016['meanAlbedo'],yerr = DZ2016['STDAlbedo'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
       
    for n, label in enumerate(axes[0,0].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[0,0].tick_params(axis='x', labelrotation= 45)

    axes[0,1].errorbar(DZ2017.Date,DZ2017['meanAlbedo'],yerr = DZ2017['STDAlbedo'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    
    for n, label in enumerate(axes[0,1].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[0,1].tick_params(axis='x', labelrotation= 45)

    axes[0,2].errorbar(DZ2018.Date,DZ2018['meanAlbedo'],yerr = DZ2018['STDAlbedo'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)

    for n, label in enumerate(axes[0,2].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[0,2].tick_params(axis='x', labelrotation= 45)

    axes[0,3].errorbar(DZ2019.Date,DZ2019['meanAlbedo'],yerr = DZ2019['STDAlbedo'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    for n, label in enumerate(axes[0,3].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[0,3].tick_params(axis='x', labelrotation= 45)

    #set y labels and limits
    axes[0,0].set_ylabel('Albedo',fontsize=16)    
    axes[0,0].set_ylim(0,1.0)
    axes[0,1].set_ylim(0,1.0)
    axes[0,2].set_ylim(0,1.0)
    axes[0,3].set_ylim(0,1.0)

    ##################
    # SUBPLOT 2: Algae

    axes[1,0].errorbar(DZ2016.Date,DZ2016['meanAlgae'],yerr = DZ2016['STDAlgae'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
       
    for n, label in enumerate(axes[1,0].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[1,0].tick_params(axis='x', labelrotation= 45)

    axes[1,1].errorbar(DZ2017.Date,DZ2017['meanAlgae'],yerr = DZ2017['STDAlgae'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    
    for n, label in enumerate(axes[1,1].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[1,1].tick_params(axis='x', labelrotation= 45)

    axes[1,2].errorbar(DZ2018.Date,DZ2018['meanAlgae'],yerr = DZ2018['STDAlgae'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)

    for n, label in enumerate(axes[1,2].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[1,2].tick_params(axis='x', labelrotation= 45)

    axes[1,3].errorbar(DZ2019.Date,DZ2019['meanAlgae'],yerr = DZ2019['STDAlgae'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    for n, label in enumerate(axes[1,3].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[1,3].tick_params(axis='x', labelrotation= 45)

    #set y labels and limits
    axes[1,0].set_ylabel('Algae',fontsize=16)    
    axes[1,0].set_ylim(0,125000)
    axes[1,1].set_ylim(0,125000)
    axes[1,2].set_ylim(0,125000)
    axes[1,3].set_ylim(0,125000)


    #######################
    # SUBPLOT 3: Grain Size

    axes[2,0].errorbar(DZ2016.Date,DZ2016['meanGrain'],yerr = DZ2016['STDGrain'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
       
    for n, label in enumerate(axes[2,0].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[2,0].tick_params(axis='x', labelrotation= 45)

    axes[2,1].errorbar(DZ2017.Date,DZ2017['meanGrain'],yerr = DZ2017['STDGrain'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    
    for n, label in enumerate(axes[2,1].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[2,1].tick_params(axis='x', labelrotation= 45)

    axes[2,2].errorbar(DZ2018.Date,DZ2018['meanGrain'],yerr = DZ2018['STDGrain'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)

    for n, label in enumerate(axes[2,2].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[2,2].tick_params(axis='x', labelrotation= 45)

    axes[2,3].errorbar(DZ2019.Date,DZ2019['meanGrain'],yerr = DZ2019['STDGrain'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    for n, label in enumerate(axes[2,3].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[2,3].tick_params(axis='x', labelrotation= 45)

    #set y labels and limits
    axes[2,0].set_ylabel('Grain Size (microns)',fontsize=16)    
    axes[2,0].set_ylim(0,20000)
    axes[2,1].set_ylim(0,20000)
    axes[2,2].set_ylim(0,20000)
    axes[2,3].set_ylim(0,20000)
    
    ####################
    # SUBPLOT 4: Density

    axes[3,0].errorbar(DZ2016.Date,DZ2016['meanDensity'],yerr = DZ2016['STDDensity'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
       
    for n, label in enumerate(axes[3,0].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[3,0].tick_params(axis='x', labelrotation= 45)

    axes[3,1].errorbar(DZ2017.Date,DZ2017['meanDensity'],yerr = DZ2017['STDDensity'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    
    for n, label in enumerate(axes[3,1].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[3,1].tick_params(axis='x', labelrotation= 45)

    axes[3,2].errorbar(DZ2018.Date,DZ2018['meanDensity'],yerr = DZ2018['STDDensity'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)

    for n, label in enumerate(axes[3,2].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[3,2].tick_params(axis='x', labelrotation= 45)

    axes[3,3].errorbar(DZ2019.Date,DZ2019['meanDensity'],yerr = DZ2019['STDDensity'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    for n, label in enumerate(axes[3,3].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[3,3].tick_params(axis='x', labelrotation= 45)


    #set y labels and limits
    axes[3,0].set_ylabel('Density (kg m-3)',fontsize=16)    
    axes[3,0].set_ylim(0,1000)
    axes[3,1].set_ylim(0,1000)
    axes[3,2].set_ylim(0,1000)
    axes[3,3].set_ylim(0,1000)



    ############################
    # SUBPLOT: COVERAGE BY CLASS
    ############################

    DZ_Class['TotalAlg'] = DZ_Class[['ClassCountHA','ClassCountLA']].sum(axis=1)

    DZ_Class2016 = DZ_Class[DZ_Class.Date.between('2016-06-01','2016-08-31')]

    DZ_Class2017 = DZ_Class[DZ_Class.Date.between('2017-06-01','2017-08-31')]
    
    DZ_Class2018 = DZ_Class[DZ_Class.Date.between('2018-06-01','2018-08-31')]
    
    DZ_Class2019 = DZ_Class[DZ_Class.Date.between('2019-06-01','2019-08-31')]

    ########
    # 2016

    axes[4,0].plot(DZ_Class2016.Date.astype(str),DZ_Class2016.ClassCountSN*0.004,label='Snow',color='y',linestyle='-.')
    axes[4,0].plot(DZ_Class2016.Date.astype(str),DZ_Class2016.ClassCountCI*0.004,label='Clean Ice',color='b')
    axes[4,0].plot(DZ_Class2016.Date.astype(str),DZ_Class2016.ClassCountLA*0.004,label='Light Algae',color='r')
    axes[4,0].plot(DZ_Class2016.Date.astype(str),DZ_Class2016.ClassCountHA*0.004,label='Heavy Algae',color='g')
    axes[4,0].plot(DZ_Class2016.Date.astype(str),DZ_Class2016.ClassCountSN*0.004,label='Total Algae',color='k',linestyle='--',alpha=0.4)    
    
    every_nth = 10
    for n, label in enumerate(axes[4,0].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[4,0].tick_params(axis='x', labelrotation= 45)

    axes[4,0].set_ylabel('Areal coverage (km^2)', fontsize=16)
    axes[4,0].set_ylim(0,50000)
    
    ########
    # 2017

    axes[4,1].plot(DZ_Class2017.Date.astype(str),DZ_Class2017.ClassCountSN*0.004,label='Snow',color='y',linestyle='-.')
    axes[4,1].plot(DZ_Class2017.Date.astype(str),DZ_Class2017.ClassCountCI*0.004,label='Clean Ice',color='b')
    axes[4,1].plot(DZ_Class2017.Date.astype(str),DZ_Class2017.ClassCountLA*0.004,label='Light Algae',color='r')
    axes[4,1].plot(DZ_Class2017.Date.astype(str),DZ_Class2017.ClassCountHA*0.004,label='Heavy Algae',color='g')
    axes[4,1].plot(DZ_Class2017.Date.astype(str),DZ_Class2017.ClassCountSN*0.004,label='Total Algae',color='k',linestyle='--',alpha=0.4)    
    axes[4,1].set_ylim(0,50000)
    every_nth = 10
    for n, label in enumerate(axes[4,1].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[4,1].tick_params(axis='x', labelrotation= 45)


    ########
    # 2018

    axes[4,2].plot(DZ_Class2018.Date.astype(str),DZ_Class2018.ClassCountSN*0.004,label='Snow',color='y',linestyle='-.')
    axes[4,2].plot(DZ_Class2018.Date.astype(str),DZ_Class2018.ClassCountCI*0.004,label='Clean Ice',color='b')
    axes[4,2].plot(DZ_Class2018.Date.astype(str),DZ_Class2018.ClassCountLA*0.004,label='Light Algae',color='r')
    axes[4,2].plot(DZ_Class2018.Date.astype(str),DZ_Class2018.ClassCountHA*0.004,label='Heavy Algae',color='g')
    axes[4,2].plot(DZ_Class2018.Date.astype(str),DZ_Class2018.ClassCountSN*0.004,label='Total Algae',color='k',linestyle='--',alpha=0.4)    
    axes[4,2].set_ylim(0,50000)
    every_nth = 10
    for n, label in enumerate(axes[4,2].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[4,2].tick_params(axis='x', labelrotation= 45)


    ########
    # 2019

    axes[4,3].plot(DZ_Class2019.Date.astype(str),DZ_Class2019.ClassCountSN*0.004,label='Snow',color='y',linestyle='-.')
    axes[4,3].plot(DZ_Class2019.Date.astype(str),DZ_Class2019.ClassCountCI*0.004,label='Clean Ice',color='b')
    axes[4,3].plot(DZ_Class2019.Date.astype(str),DZ_Class2019.ClassCountLA*0.004,label='Light Algae',color='r')
    axes[4,3].plot(DZ_Class2019.Date.astype(str),DZ_Class2019.ClassCountHA*0.004,label='Heavy Algae',color='g')
    axes[4,3].plot(DZ_Class2019.Date.astype(str),DZ_Class2019.ClassCountSN*0.004,label='Total Algae',color='k',linestyle='--',alpha=0.4)    
    axes[4,3].set_ylim(0,50000)
    every_nth = 10
    for n, label in enumerate(axes[4,3].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    axes[4,3].tick_params(axis='x', labelrotation= 45)


    # Add text labels to columns for years
    axes[0,0].text('2016-07-01',1.05,'2016',fontsize=26)
    axes[0,1].text('2017-07-01',1.05,'2017',fontsize=26)
    axes[0,2].text('2018-07-01',1.05,'2018',fontsize=26)
    axes[0,3].text('2019-07-01',1.05,'2019',fontsize=26) 

    # tight layout and save figure
    fig.tight_layout()

    plt.savefig(str(savepath+'Multipanel_BISC.png'))

    return



def correlate_vars(DZ, savepath):

    """
    Function scatters each variable against all others and presents as a multipanel figure.
    Linear regression line, r2, p and Pearson's R are reported in each panel.

    params:
        - DZ: pandas dataframe time series for each variable across the dark zone 
        - savepath: path to save figures to
    returns:
        -None
    """

    fig, ax = plt.subplots(3,2, figsize=(15,20))
    
    #########################
    # SUBPLOT 1: Albedo/Algae
    #########################
    X = DZ.meanAlbedo
    Y = DZ.meanAlgae
    ax[0,0].scatter(X,Y, marker='x', color = 'b')
    ax[0,0].set_xlabel('Albedo',fontsize=20)
    ax[0,0].set_ylabel('Algae (ppb)',fontsize=20)

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[0,0].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[0,0].text(0.7,60000,"r2 = {}\nR = {}\np={}".format(abs(np.round(r_value**2,2)),\
        np.round(pearson_coef,2),np.round(pearson_p,6)),fontsize=16)

    ##############################
    # SUBPLOT 2: Albedo/Grain Size
    ##############################
    X = DZ.meanAlbedo
    Y = DZ.meanGrain
    ax[0,1].scatter(X,Y, marker='x', color = 'b')
    ax[0,1].set_xlabel('Albedo', fontsize=20)
    ax[0,1].set_ylabel('Grain size (microns)',fontsize=20)

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[0,1].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[0,1].text(0.7,12000,"r2 = {}\nR = {}\np={}".format(abs(np.round(r_value**2,2)),\
        np.round(pearson_coef,2),np.round(pearson_p,6)),fontsize=16)


    ###########################
    # SUBPLOT 3: Albedo/Density
    ###########################
    X = DZ.meanAlbedo
    Y = DZ.meanDensity
    ax[1,0].scatter(X,Y, marker='x', color = 'b')
    ax[1,0].set_xlabel('Albedo',fontsize=20)
    ax[1,0].set_ylabel('Density (kg m-3)',fontsize=20)

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[1,0].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[1,0].text(0.7,450,"r2 = {}\nR = {}\np={}".format(abs(np.round(r_value**2,2)),\
        np.round(pearson_coef,2),np.round(pearson_p,6)),fontsize=16)

    #############################
    # SUBPLOT 4: Algae/Grain Size
    #############################
    X = DZ.meanAlgae
    Y = DZ.meanGrain
    ax[1,1].scatter(X,Y, marker='x', color = 'b')
    ax[1,1].set_xlabel('Algae (ppb)',fontsize=20)
    ax[1,1].set_ylabel('Grain size (microns)',fontsize=20)

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[1,1].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[1,1].text(60000, 5750,"r2 = {}\nR = {}\np={}".format(abs(np.round(r_value**2,2)),\
        np.round(pearson_coef,2),np.round(pearson_p,6)),fontsize=16)

    ##########################
    # SUBPLOT 7: Algae/Density
    ##########################
    X = DZ.meanAlgae
    Y = DZ.meanDensity 
    ax[2,0].scatter(X,Y, marker='x', color = 'b')
    ax[2,0].set_xlabel('Algae (ppb)',fontsize=20)
    ax[2,0].set_ylabel('Density (kg m-3)',fontsize=20)

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[2,0].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[2,0].text(60000,550,"r2 = {}\nR = {}\np={}".format(abs(np.round(r_value**2,2)),\
        np.round(pearson_coef,2),np.round(pearson_p,6)),fontsize=16)


    ################################
    # SUBPLOT 10: Grain Size/Density
    ################################
    X = DZ.meanGrain
    Y = DZ.meanDensity 
    ax[2,1].scatter(X,Y, marker='x', color = 'b')
    ax[2,1].set_xlabel('Grain Size (microns)',fontsize=20)
    ax[2,1].set_ylabel('Density (kg m-3)',fontsize=20)

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[2,1].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[2,1].text(3500,435,"r2 = {}\nR = {}\np={}".format(abs(np.round(r_value**2,2)),\
    np.round(pearson_coef,2),np.round(pearson_p,6)),fontsize=16)

    ax[2,1].set_xlim(2500,14000)

    fig.tight_layout()
    plt.savefig(str(savepath+'correlations.png'),dpi=300)
    plt.show()
    
    return


def annual_trends(DZ, savepath):
    """
    Function creates a 6 panel figure. In each of panels 1-5, a 5day rolling mean 
    for a variable is plotted with a separate line for each year. The linear
    regression line is also plotted for each variable/year in the matching colour.
    Regression statistics are reported in the legend of each subpanel.

    In panel 6, a heatmap of the correlations between each variable in DZ is plotted.

    params:
        DZ: pandas dataframe containing time series for each variable for the whole dark zone
        savepath: path to save figures to
    
    returns:
        - None
    """

    fig, ax = plt.subplots(3,3, figsize=(23,15))

    X1 = DZ['Date'].loc[DZ['Date'].dt.year==2016]
    X2 = DZ['Date'].loc[DZ['Date'].dt.year==2017]
    X3 = DZ['Date'].loc[DZ['Date'].dt.year==2018]
    X4 = DZ['Date'].loc[DZ['Date'].dt.year==2019]

    XX1 = DZ_Class['Date'].loc[DZ_Class['Date'].dt.year==2016]
    XX2 = DZ_Class['Date'].loc[DZ_Class['Date'].dt.year==2017]
    XX3 = DZ_Class['Date'].loc[DZ_Class['Date'].dt.year==2018]
    XX4 = DZ_Class['Date'].loc[DZ_Class['Date'].dt.year==2019]   
    colours = ['b','g','k','y']
    

    ##########
    # Albedo
    ##########

    Y1 = DZ['meanAlbedo'].loc[DZ['Date'].dt.year==2016]
    Y2 = DZ['meanAlbedo'].loc[DZ['Date'].dt.year==2017]
    Y3 = DZ['meanAlbedo'].loc[DZ['Date'].dt.year==2018]
    Y4 = DZ['meanAlbedo'].loc[DZ['Date'].dt.year==2019]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]

    years = ['2016','2017','2018','2019']
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[0,0].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[0,0].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    
    ax[0,0].legend(loc='best')
    ax[0,0].set_xticks([],[])
    ax[0,0].set_ylabel('Albedo', fontsize=12)
    ax[0,0].set_ylim(0.3,0.9)


    ########
    # Algae
    ########

    Y1 = DZ['meanAlgae'].loc[DZ['Date'].dt.year==2016]
    Y2 = DZ['meanAlgae'].loc[DZ['Date'].dt.year==2017]
    Y3 = DZ['meanAlgae'].loc[DZ['Date'].dt.year==2018]
    Y4 = DZ['meanAlgae'].loc[DZ['Date'].dt.year==2019]
    
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[0,1].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[0,1].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    
    ax[0,1].set_xticks([],[])
    ax[0,1].set_ylim(0,90000)
    ax[0,1].set_ylabel('Algae (ppb)', fontsize=12)
    ax[0,1].legend(loc='best')

    ############
    # Grain Size
    ############

    Y1 = DZ['meanGrain'].loc[DZ['Date'].dt.year==2016]
    Y2 = DZ['meanGrain'].loc[DZ['Date'].dt.year==2017]
    Y3 = DZ['meanGrain'].loc[DZ['Date'].dt.year==2018]
    Y4 = DZ['meanGrain'].loc[DZ['Date'].dt.year==2019]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[0,2].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[0,2].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    
    ax[0,2].set_xticks([],[])
    ax[0,2].set_ylim(0,25000)
    ax[0,2].set_ylabel('Grain Size (microns)', fontsize=12)
    ax[0,2].legend(loc='best')
    

    ##########
    # Density
    ##########

    Y1 = DZ['meanDensity'].loc[DZ['Date'].dt.year==2016]
    Y2 = DZ['meanDensity'].loc[DZ['Date'].dt.year==2017]
    Y3 = DZ['meanDensity'].loc[DZ['Date'].dt.year==2018]
    Y4 = DZ['meanDensity'].loc[DZ['Date'].dt.year==2019]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[1,0].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[1,0].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))

    ax[1,0].set_xticks([],[])
    ax[1,0].set_ylim(400,700)
    ax[1,0].set_ylabel('Density (kg m-3)', fontsize=12)
    ax[1,0].legend(loc='best')


    ###############
    # SNOW COVERAGE
    ###############
    DZ_Class['SnowArea'] = DZ_Class.ClassCountSN * 0.0004
    DZ_Class['HAArea'] = DZ_Class.ClassCountHA * 0.0004
    DZ_Class['LAArea'] = DZ_Class.ClassCountLA * 0.0004
    DZ_Class['CIArea'] = DZ_Class.ClassCountCI * 0.0004
    DZ_Class['CCArea'] = DZ_Class.ClassCountCC * 0.0004
    DZ_Class['WATArea'] = DZ_Class.ClassCountWAT * 0.0004

    Y1 = DZ_Class['SnowArea'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['SnowArea'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['SnowArea'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['SnowArea'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[1,1].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[1,1].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))

    ax[1,1].set_xticks([],[])
    ax[1,1].set_ylim(0,5000)
    ax[1,1].set_ylabel('Area Covered by SNOW (km^2)', fontsize=12)
    ax[1,1].legend(loc='best')


    ###############
    # CI COVERAGE
    ###############


    Y1 = DZ_Class['CIArea'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['CIArea'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['CIArea'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['CIArea'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[1,2].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[1,2].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))

    ax[1,2].set_xticks([],[])
    ax[1,2].set_ylim(1000,6000)
    ax[1,2].set_ylabel('Area Covered by CLEAN ICE (km^2)', fontsize=12)
    ax[1,2].legend(loc='best')


    ###############
    # HA COVERAGE
    ###############

    Y1 = DZ_Class['HAArea'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['HAArea'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['HAArea'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['HAArea'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[2,0].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[2,0].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))

    date_labels = []
  
    for d in X:
        date_labels.append("{}-0{}".format(d.day,d.month))

    print(date_labels)

    ax[2,0].set_xticklabels(date_labels[::4], rotation=45, fontsize=10)

    ax[2,0].set_ylim(0,700)
    ax[2,0].set_ylabel('Area Covered by HEAVY ALGAE (km^2)', fontsize=12)
    ax[2,0].set_xlabel('Date (day-month)', fontsize=12)
    ax[2,0].legend(loc='best')


    ###############
    # LA COVERAGE
    ###############

    Y1 = DZ_Class['LAArea'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['LAArea'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['LAArea'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['LAArea'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[2,1].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[2,1].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))


    ax[2,1].set_xticklabels(date_labels[::4], rotation=45, fontsize=10)
    ax[2,1].set_ylim(0,3000)
    ax[2,1].set_ylabel('Area Covered by LIGHT ALGAE (km^2)', fontsize=12)
    ax[2,1].set_xlabel('Date (day-month)', fontsize=12)
    ax[2,1].legend(loc='best')


    ##############
    # WAT COVERAGE
    ##############

    Y1 = DZ_Class['WATArea'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['WATArea'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['WATArea'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['WATArea'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:28]
        Y = Ys[i][0:28]
        Xnums = np.arange(0,len(X),1)
        ax[2,2].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[2,2].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))

    ax[2,2].set_xticklabels(date_labels[::4], rotation=45, fontsize=10)
    ax[2,2].set_ylim(0,50)
    ax[2,2].set_ylabel('Area Covered by WATER (km^2)',fontsize=12)
    ax[2,2].set_xlabel('Date (day-month)', fontsize=12)
    ax[2,2].legend(loc='best')
    
    fig.tight_layout()
    
    fig.savefig(str(savepath+'/JJA_trends.png'),dpi=300)

    return


def Latitudinal_trends(DZ, wea,web,wec,wev,weu,wet, savepath,\
     anova_assumptions = False, KW_test = True):
    
    """
    
     try: compare JJA means for each tile - organise into N-S order and test for trend
     try: statistical difference test for JJA time series for each tile
     try: time series comparison

    """

    def test_anova_assumptions():
 
        # check ANOVA assumptions
        # does the data meet criteria for variance, independence and normality
        
        var = np.var(wea.meanAlbedo) # find variance of one column and use as target
        normAlpha = 0.05 # set critical value

        for i in [wea,web,wec,wet,weu,wev]: # loop through columns, compare to target
            
            NormStat,Norm_p = shapiro(i.meanAlbedo) # shapiro test for normality
            variance = np.var(i.meanAlbedo) # variance of each column

            if Norm_p > normAlpha:
                
                print("Shapiro walk test: stat = {}, p = {}".format(NormStat,Norm_p))
                print("NORMALITY ASSUMPTION SATISFIED")
            
            else:
                print("Shapiro walk test: stat = {}, p = {}".format(NormStat,Norm_p))
                print('ANOVA NORMALITY ASSUMPTION VIOLATED')

            if np.round(abs(var-variance),2) > 0.01:
                print('EQUAL VARIANCE ASSUMPTION VIOLATED')
            else:
                print("EQUAL VARIANCE ASSUMPTION SATISFIED")
    
        # Chi square test for sample independence
        independenceDF =pd.DataFrame(columns = ['wea','web','wec','wet','weu','wev'])
        independenceDF['wea'] = wea.meanAlbedo
        independenceDF['web'] = web.meanAlbedo
        independenceDF['wec'] = wec.meanAlbedo
        independenceDF['wet'] = wet.meanAlbedo
        independenceDF['weu'] = weu.meanAlbedo
        independenceDF['wev'] = wev.meanAlbedo
        
        Chi_stat, Chi_p, dof, expected = chi2_contingency(independenceDF)
        alpha = 1.0 - Chi_p

        if Chi_p<= alpha:
            print("Chi2 stat = {}, Chi2 p-value = {}, DOF = {}".format(Chi_stat,Chi_p, dof))
            print('INDEPENDENCE ASSUMPTION VIOLATED')
        else:
            print("Chi2 stat = {}, Chi2 p-value = {}, DOF = {}".format(Chi_stat,Chi_p, dof))
            print("INDEPENDENCE ASSUMPTION SATISFIED")
        
        return

     
    def KruskalWallis():

        year = 2019
        kruskalDF =pd.DataFrame(columns = ['Date','wea','web','wec','wet','weu','wev'])
        kruskalDF.Date = DZ.Date
        kruskalDF['wea'] = wea.meanAlgae
        kruskalDF['web'] = web.meanAlgae
        kruskalDF['wec'] = wec.meanAlgae
        kruskalDF['wet'] = wet.meanAlgae
        kruskalDF['weu'] = weu.meanAlgae
        kruskalDF['wev'] = wev.meanAlgae

        Col1 = kruskalDF['wea'].loc[kruskalDF['Date'].dt.year==year]
        Col2 = kruskalDF['web'].loc[kruskalDF['Date'].dt.year==year]
        Col3 = kruskalDF['wec'].loc[kruskalDF['Date'].dt.year==year]
        Col4 = kruskalDF['wet'].loc[kruskalDF['Date'].dt.year==year]
        Col5 = kruskalDF['weu'].loc[kruskalDF['Date'].dt.year==year]
        Col6 = kruskalDF['wev'].loc[kruskalDF['Date'].dt.year==year]

        H, pval = mstats.kruskalwallis([Col1, Col2, Col3, Col4, Col5, Col6])
        
        print("\nKRUSKAL_WALLIS TEST")
        print("H-statistic:", H)
        print("P-Value:", pval)
        

        if pval < 0.05:
            print("Reject NULL hypothesis - Significant differences exist between groups.")
        if pval > 0.05:
            print("Accept NULL hypothesis - No significant difference between groups.")
        

        return

    if anova_assumptions:
        test_anova_assumptions()

    if KW_test:
        KruskalWallis()

    
    return

def plot_coverage(DZ, DZ_Class,savepath):

    DZ_Class['TotalAlg'] = DZ_Class[['ClassCountHA','ClassCountLA']].sum(axis=1)

    fig, ax = plt.subplots(1,1,figsize=(15,10))
    ax.set_ylim(0,1.45e7)
    ax.plot(DZ_Class.Date.astype(str),DZ_Class.ClassCountSN,label='Snow',color='y',linestyle='-.')
    ax.plot(DZ_Class.Date.astype(str),DZ_Class.ClassCountCI,label='Clean Ice',color='b')
    ax.plot(DZ_Class.Date.astype(str),DZ_Class.ClassCountLA,label='Light Algae',color='r')
    ax.plot(DZ_Class.Date.astype(str),DZ_Class.ClassCountHA,label='Heavy Algae', color='g')
    ax.plot(DZ_Class.Date.astype(str),DZ_Class.TotalAlg,label='Total Algae',color='k',linestyle='--',alpha=0.4)
    ax.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)

    ax.text('2016-07-01',1.25e7,'2016',fontsize=22)
    ax.text('2017-07-01',1.25e7,'2017',fontsize=22)
    ax.text('2018-07-01',1.25e7,'2018',fontsize=22)
    ax.text('2019-07-01',1.25e7,'2019',fontsize=22)

    ax.set_ylabel('N pixels x 1e7',fontsize=16)
    ax.set_xlabel('Date',fontsize=16)

    every_nth = 10
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.tick_params(axis='x', labelrotation= 45)

    ax.legend(loc='upper right', ncol=5,fontsize=16)

    fig.savefig(str(savepath+'SpatialCoverageTimesSeries.png'),dpi=300)
    
    return

def corr_heatmaps(DZ,DZ_Class,savepath):

    DZCorr = DZ[['meanAlbedo','meanGrain','meanDensity','meanAlgae','DOY']]
    DZCorr.columns = [['Albedo','Grain Size','Density','Algae','DOY']]
    
    DZ_Class_Corr = DZ_Class[['ClassCountSN','ClassCountCI','ClassCountHA','ClassCountLA','ClassCountWAT','ClassCountCC']]
    DZ_Class_Corr.columns=[['Snow','Clean Ice','Heavy Algae','Light Algae','Water','Cryoconite']]
    DZ_Class_Corr['TotalAlg'] = DZ_Class[['ClassCountHA','ClassCountLA']].sum(axis=1)

    fig, ax = plt.subplots(1,1,figsize=(20,8))
    ax[0] = sns.heatmap(DZCorr.corr(),annot=True,cmap='coolwarm',ax=ax[0])
    ax[0].set_title('RT variables')
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)

    # ax[1] = sns.heatmap(DZ_Class_Corr.corr(),annot=True,cmap='coolwarm',ax=ax[1])
    # ax[1].set_title('Spatial Coverage')
    # ax[1].set_ylabel(None)
    # ax[1].set_xlabel(None)
    # fig.savefig(str(savepath+'Heatmaps.png'),dpi=300)

    return

def model_fits(var,degree, start, stop):
    """
    params:

    - start, stop: index number for start and end of defined range
    - degree: degree of polynomial to fit
    - var: which variable to fit - pass DZ.var

    returns:
    
    """
    x = np.arange(start,stop,1)

    y = var[start:stop]

    coeffs = np.polyfit(x, y, 2)
    model = np.poly1d(coeffs)
    results = smf.ols(formula='y ~ model(x)', data=var).fit()

    linresults = stats.linregress(x,y)

    return coeffs, results, linresults

def compare_vars_between_classes(DZ_Class):

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1, figsize=(20,10))

    ax1.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.AlgaeMeanHA, color='g', label='HA', marker = 'x')
    ax1.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.AlgaeMeanLA, color='r', label='LA', marker = 'x')
    ax1.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.AlgaeMeanCI, color='b', label='CI', marker = 'x')
    ax1.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.AlgaeMeanCC, color='k', label='CC', marker = 'x')
    ax1.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.AlgaeMeanWAT, color='grey', label = 'WAT',marker = 'x')
    ax1.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.AlgaeMeanSN, color='y', label='SN',marker = 'x')
    ax1.set_ylabel('Algae ppb')
    ax1.set_xticks([])
    
    ax2.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.GrainMeanHA, color='g', label='HA', marker = 'x')
    ax2.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.GrainMeanLA, color='r', label='LA', marker = 'x')
    ax2.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.GrainMeanCI, color='b', label='CI', marker = 'x')
    ax2.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.GrainMeanCC, color='k', label='CC', marker = 'x')
    ax2.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.GrainMeanWAT, color='grey', label='WAT', marker = 'x')
    ax2.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.GrainMeanSN, color='y', label='SN', marker = 'x')
    ax2.set_ylabel('Grian size (microns)')
    ax2.set_xticks([])

    ax3.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.DensityMeanHA, color='g', label='HA', marker = 'x')
    ax3.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.DensityMeanLA, color='r', label='LA', marker = 'x')
    ax3.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.DensityMeanCI, color='b', label='CI', marker = 'x')
    ax3.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.DensityMeanCC, color='k', label='CC', marker = 'x')
    ax3.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.DensityMeanWAT, color='grey', label='WAT', marker = 'x')
    ax3.scatter(np.arange(0,len(DZ_Class),1), DZ_Class.DensityMeanSN, color='y', label='SN', marker = 'x')
    ax3.set_ylabel('Density (kg m-3)')
    ax1.legend(loc='upper right')

    return



def lin_regress():

    import statsmodels.api as sm
    import pandas as pd

    DZ = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/WholeDZ.csv')
    y = DZ['meanAlbedo']
    X = DZ[['meanDensity(kgm-3)','meanGrain(um)']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())

    return


# Function Calls

filepath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/'
savepath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Figures_and_Tables/'

DZ, DZ_Class, wea, web, wec, wet, weu, wev, BandRatios = import_csvs(filepath,True,True,True)
#timeseries_STDerrorbars(DZ, wea, web, wec, wet, weu, wev, savepath)

# correlate_vars(DZ, savepath)
# corr_heatmaps(DZ,DZ_Class,savepath)
# annual_trends(DZ,savepath)
# Latitudinal_trends(DZ, wea,web,wec,wev,weu,wet, savepath,\
# anova_assumptions = True, KW_test = True)
# plot_coverage(DZ, DZ_Class,savepath)
# compare_vars_between_classes(DZ_Class)


plot_BandRatios(BandRatios, savepath)


# import statsmodels.api as sm
# import pandas as pd

# DZ = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/WholeDZ.csv')
# y = DZ['meanAlbedo']
# X = DZ[['meanDensity(kgm-3)','meanGrain(um)']]
# X = sm.add_constant(X)
# model = sm.OLS(y, X)
# results = model.fit()
# print(results.summary())

