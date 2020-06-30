import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime
import scipy 
from scipy import stats # For in-built method to get PCC


def import_csvs(DZ = True, Tiles = True, filepath):
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
        DZ = pd.read_csv(str(filepath+'WholeDZ.csv')
        DZ = DZ.sort_values(by='Date').reset_index(drop=True)

    else:
         DZ = None

    if Tiles:

        # import individual tiles
        wea = pd.read_csv(str(filepath+'/22wea.csv'))
        web = pd.read_csv(str(filepath+'/22wea.csv'))
        wec = pd.read_csv(str(filepath+'/22wea.csv'))
        wet = pd.read_csv(str(filepath+'/22wea.csv'))
        weu = pd.read_csv(str(filepath+'/22wea.csv'))
        wev = pd.read_csv(str(filepath+'/22wea.csv'))

    else: 
        wea = None
        web =None
        wec = None
        wet = None
        weu = None
        wev = None

    return DZ, wea, web, wec, wet, weu, wev



def AllTileVars(DZ, wea, web, wec, wet, weu, wev, savepath):

    """
    
    function plots the time series for each variable (albedo, grain size, density, dust, algae...)
    in separate subplots in a 5 panel figure. The values for the total DZ and for each individual 
    tile are plotted on the same axes for comparison.

    params:
        - DZ: pandas dataframe containing time series for each variable across the whole dark zone
        - wea...wev: pandas dataframes containing time series for each variable for each individual tile
        - savepath: path to save figure to
    returns:
        - None

    """
    # 1) How do the variables change over time?
    # solid black line for whole DZ, coloured lines for individual tiles

    fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize = (20,15))
    
    ax1.text('2016-07-01',1.3,'2016',fontsize=22)
    ax1.text('2017-07-01',1.3,'2017',fontsize=22)
    ax1.text('2018-07-01',1.3,'2018',fontsize=22)
    ax1.text('2019-07-01',1.3,'2019',fontsize=22)

    # SUBPLOT 1: Albedo
    ax1.plot(DZ['Date'],DZ.meanAlbedo, label = "DZ mean",color='k')
    ax1.plot(DZ['Date'],wea.meanAlbedo,label="22WEA",color='k',alpha=0.4)
    ax1.plot(DZ['Date'],web.meanAlbedo,label="22WEB",color='b',alpha=0.4)
    ax1.plot(DZ['Date'],wec.meanAlbedo,label="22WEC",color='g',alpha=0.4)
    ax1.plot(DZ['Date'],wet.meanAlbedo,label="22WET",color='y',alpha=0.4)
    ax1.plot(DZ['Date'],weu.meanAlbedo,label="22WEU",color='g',linestyle = '--', alpha=0.4)
    ax1.plot(DZ['Date'],wev.meanAlbedo,label="22WEV",color='k',linestyle = '--',alpha=0.4)
    ax1.set_ylim(0,1.2)
    ax1.set_xticks([], [])

    ax1.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax1.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax1.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax1.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)

    ax1.legend(loc='upper right',ncol=7, fontsize=16)
    ax1.set_ylabel('Albedo',fontsize=20)

    # SUBPLOT 2: Algae
    ax2.plot(DZ['Date'],DZ.meanAlgae, label = "DZ mean",color='k')
    ax2.plot(DZ['Date'],wea.meanAlgae,label="22WEA",color='k',alpha=0.4)
    ax2.plot(DZ['Date'],web.meanAlgae,label="22WEB",color='b',alpha=0.4)
    ax2.plot(DZ['Date'],wec.meanAlgae,label="22WEC",color='g',alpha=0.4)
    ax2.plot(DZ['Date'],wet.meanAlgae,label="22WET",color='y',alpha=0.4)
    ax2.plot(DZ['Date'],weu.meanAlgae,label="22WEU",color='g',linestyle = '--', alpha=0.4)
    ax2.plot(DZ['Date'],wev.meanAlgae,label="22WEV",color='k',linestyle = '--',alpha=0.4)

    ax2.set_xticks([],[])

    ax2.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax2.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax2.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax2.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)

    ax2.set_ylabel('Algae (ppb)',fontsize=16)
    
    # SUBPLOT 3: Grain Size
    ax3.plot(DZ['Date'],DZ.meanGrain, label = "DZ mean",color='k')
    ax3.plot(DZ['Date'],wea.meanGrain,label="22WEA",color='k',alpha=0.4)
    ax3.plot(DZ['Date'],web.meanGrain,label="22WEB",color='b',alpha=0.4)
    ax3.plot(DZ['Date'],wec.meanGrain,label="22WEC",color='g',alpha=0.4)
    ax3.plot(DZ['Date'],wet.meanGrain,label="22WET",color='y',alpha=0.4)
    ax3.plot(DZ['Date'],weu.meanGrain,label="22WEU",color='g',linestyle = '--', alpha=0.4)
    ax3.plot(DZ['Date'],wev.meanGrain,label="22WEV",color='k',linestyle = '--',alpha=0.4)

    ax3.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax3.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax3.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax3.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)

    ax3.set_xticks([],[])
    ax3.set_ylabel('Ice grain size (microns)',fontsize=16)
    
    # SUBPLOT 4: Density
    ax4.plot(DZ['Date'],DZ.meanDensity, label = "DZ mean",color='k')
    ax4.plot(DZ['Date'],wea.meanDensity,label="22WEA",color='k',alpha=0.4)
    ax4.plot(DZ['Date'],web.meanDensity,label="22WEB",color='b',alpha=0.4)
    ax4.plot(DZ['Date'],wec.meanDensity,label="22WEC",color='g',alpha=0.4)
    ax4.plot(DZ['Date'],wet.meanDensity,label="22WET",color='y',alpha=0.4)
    ax4.plot(DZ['Date'],weu.meanDensity,label="22WEU",color='g',linestyle = '--', alpha=0.4)
    ax4.plot(DZ['Date'],wev.meanDensity,label="22WEV",color='k',linestyle = '--',alpha=0.4)

    ax4.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax4.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax4.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax4.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)

    ax4.set_ylabel("Ice Density (kg m-3)",fontsize=16)
    ax4.set_xticks([],[])

    # SUBPLOT 5: Dust
    ax5.plot(DZ['Date'],DZ.meanDust, label = "DZ mean",color='k')
    ax5.plot(DZ['Date'],wea.meanDust,label="22WEA",color='k',alpha=0.4)
    ax5.plot(DZ['Date'],web.meanDust,label="22WEB",color='b',alpha=0.4)
    ax5.plot(DZ['Date'],wec.meanDust,label="22WEC",color='g',alpha=0.4)
    ax5.plot(DZ['Date'],wet.meanDust,label="22WET",color='y',alpha=0.4)
    ax5.plot(DZ['Date'],weu.meanDust,label="22WEU",color='g',linestyle = '--', alpha=0.4)
    ax5.plot(DZ['Date'],wev.meanDust,label="22WEV",color='k',linestyle = '--',alpha=0.4)

    every_nth = 5
    for n, label in enumerate(ax5.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax5.tick_params(axis='x', labelrotation= 45)

    ax5.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax5.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax5.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax5.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)

    ax5.set_xlabel('Date',fontsize=16),ax5.set_ylabel("Dust (ppb)",fontsize=16)

    fig.tight_layout()
    plt.savefig(str(savepath+'Multipanel_BISC.png'))


    return

def DZvars_with_errorbars(DZ, savepath):

    """
    Function plots the times series for each variable for the total dark zone. Points are the
    mean across the dark zone at each time point and the standard deviation is plotted as grey
    error bars.

    params: 
        - DZ: pandas dataframe containing variable values for the total dark zone
        - savepath: path to save figures to
    returns:
        - None 
    """
    fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize = (20,15))
    
    ax1.text('2016-07-01',1.3,'2016',fontsize=22)
    ax1.text('2017-07-01',1.3,'2017',fontsize=22)
    ax1.text('2018-07-01',1.3,'2018',fontsize=22)
    ax1.text('2019-07-01',1.3,'2019',fontsize=22)

    # SUBPLOT 1: Albedo
    
    ax1.errorbar(DZ.Date,DZ.meanAlbedo,yerr = DZ['STDAlbedo'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    ax1.set_ylim(0,1.2)
    ax1.set_xticks([], [])

    ax1.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax1.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax1.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax1.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)
    ax1.set_xticks([], [])

    ax1.set_ylabel('Albedo',fontsize=20)


    #SUBPLOT 2: Algae
    ax2.errorbar(DZ.Date,DZ.meanAlgae,yerr = DZ['STDAlgae'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    
    ax2.set_xticks([], [])

    ax2.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax2.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax2.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax2.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)
    ax2.set_xticks([], [])

    ax2.set_ylabel('Algae (ppb)',fontsize=20)
    ax2.set_ylim(0,40000)

    #SUBPLOT 3: Grain size
    ax3.errorbar(DZ.Date,DZ.meanGrain,yerr = DZ['STDGrain'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    
    ax3.set_xticks([], [])

    ax3.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax3.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax3.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax3.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)
    ax3.set_xticks([], [])

    ax3.set_ylabel('Grain size (microns)',fontsize=20)
    ax3.set_ylim(0,20000)


    #SUBPLOT 4: Density
    ax4.errorbar(DZ.Date,DZ.meanDensity,yerr = DZ['STDDensity'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)
    
    ax4.set_xticks([], [])

    ax4.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax4.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax4.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax4.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)
    ax4.set_xticks([], [])

    ax4.set_ylabel('Density (kg m-3)',fontsize=20)
    ax4.set_ylim(0,1000)


    #SUBPLOT 5: Dust 
    ax5.errorbar(DZ.Date,DZ.meanDust,yerr = DZ['STDDust'], fmt='x', color='black',
                    ecolor='lightgray', elinewidth=1, capsize=0)

    ax5.axvspan('2016-06-01','2016-08-30',color='k',alpha=0.1)
    ax5.axvspan('2017-06-01','2017-08-30',color='red',alpha=0)
    ax5.axvspan('2018-06-01','2018-08-30',color='k',alpha=0.1)
    ax5.axvspan('2019-06-01','2019-08-30',color='red',alpha=0)
    ax5.set_ylabel('Dust (ppb)',fontsize=20)
    ax5.set_ylim(0,50000)

    every_nth = 5
    for n, label in enumerate(ax5.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax5.tick_params(axis='x', labelrotation= 45)

    ax5.set_xlabel('Date',fontsize=16),ax5.set_ylabel("Dust (ppb)",fontsize=16)

    fig.tight_layout()

    plt.savefig(str(savepath+'Multipanel_DZ_STD.png'),dpi=300)

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

    fig, ax = plt.subplots(5,2, figsize=(15,20))
    
    #########################
    # SUBPLOT 1: Albedo/Algae
    #########################
    X = DZ.meanAlbedo
    Y=DZ.meanAlgae
    ax[0,0].scatter(X,Y, marker='x', color = 'b')
    ax[0,0].set_xlabel('Albedo'),ax[0,0].set_ylabel('Algae (ppb)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[0,0].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[0,0].text(0.7,7000,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
        np.round(pearson_coef,2),np.round(pearson_p,5)))

    ##############################
    # SUBPLOT 2: Albedo/Grain Size
    ##############################
    X = DZ.meanAlbedo
    Y=DZ.meanGrain
    ax[0,1].scatter(X,Y, marker='x', color = 'b')
    ax[0,1].set_xlabel('Albedo'),ax[0,1].set_ylabel('Grain size (microns)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[0,1].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[0,1].text(0.7,10000,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
        np.round(pearson_coef,2),np.round(pearson_p,5)))


    ###########################
    # SUBPLOT 3: Albedo/Density
    ###########################
    X = DZ.meanAlbedo
    Y = DZ.meanDensity
    ax[1,0].scatter(X,Y, marker='x', color = 'b')
    ax[1,0].set_xlabel('Albedo'),ax[1,0].set_ylabel('Density (kg m-3')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[1,0].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[1,0].text(0.7,620,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
        np.round(pearson_coef,2),np.round(pearson_p,5)))

    ########################
    # SUBPLOT 4: Albedo/Dust
    ########################
    X = DZ.meanAlbedo
    Y = DZ.meanDust 
    ax[1,1].scatter(X,Y, marker='x', color = 'b')
    ax[1,1].set_xlabel('Albedo'),ax[1,1].set_ylabel('Dust (ppb)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[1,1].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[1,1].text(0.7,25000,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
        np.round(pearson_coef,2),np.round(pearson_p,5)))

    #######################
    # SUBPLOT 5: Algae/Dust
    #######################
    X = DZ.meanAlgae
    Y = DZ.meanDust
    ax[2,0].scatter(X,Y, marker='x', color = 'b')
    ax[2,0].set_xlabel('Algae (ppb)'),ax[2,0].set_ylabel('Dust (ppb)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[2,0].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[2,0].text(9000,25000,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
        np.round(pearson_coef,2),np.round(pearson_p,5)))

    #############################
    # SUBPLOT 6: Algae/Grain Size
    #############################
    X = DZ.meanAlgae
    Y = DZ.meanGrain
    ax[2,1].scatter(X,Y, marker='x', color = 'b')
    ax[2,1].set_xlabel('Algae (ppb)'),ax[2,1].set_ylabel('Grain size (microns)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[2,1].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[2,1].text(9000,5000,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
        np.round(pearson_coef,2),np.round(pearson_p,5)))

    ##########################
    # SUBPLOT 7: Algae/Density
    ##########################
    X = DZ.meanAlgae
    Y = DZ.meanDensity 
    ax[3,0].scatter(X,Y, marker='x', color = 'b')
    ax[3,0].set_xlabel('Algae (ppb)'),ax[3,0].set_ylabel('Density (kg m-3)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[3,0].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[3,0].text(9000,700,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
        np.round(pearson_coef,2),np.round(pearson_p,5)))

    ############################
    # SUBPLOT 8: Dust/Grain Size
    ############################
    X= DZ.meanDust
    Y = DZ.meanGrain 
    ax[3,1].scatter(X,Y, marker='x', color = 'b')
    ax[3,1].set_xlabel('Dust (ppb)'),ax[3,1].set_ylabel('Grain size (microns)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[3,1].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[3,1].text(30000,11000,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
    np.round(pearson_coef,2),np.round(pearson_p,5)))

    #########################
    # SUBPLOT 9: Dust/Density
    #########################
    X = DZ.meanDust
    Y = DZ.meanDensity 
    ax[4,0].scatter(X,Y, marker='x', color = 'b')
    ax[4,0].set_xlabel('Dust (ppb)'),ax[4,0].set_ylabel('Density (kg m-3)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[4,0].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[4,0].text(35000,600,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
    np.round(pearson_coef,2),np.round(pearson_p,5)))

    ################################
    # SUBPLOT 10: Grain Size/Density
    ################################
    X = DZ.meanGrain
    Y = DZ.meanDensity 
    ax[4,1].scatter(X,Y, marker='x', color = 'b')
    ax[4,1].set_xlabel('Grain Size (microns)'),ax[4,1].set_ylabel('Density (kg m-3)')

    # calculate and plot trend line/linear regression metrics and pearsons R
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
    pearson_coef, pearson_p = stats.pearsonr(X, Y)
    ax[4,1].plot(X, intercept + slope * X, color='k', linestyle='-', linewidth=0.5)
    ax[4,1].text(3500,600,"r2 = {}\np = {}\n\nR = {}\np={}".format(abs(np.round(r_value**2,2)),np.round(p_value,5),\
    np.round(pearson_coef,2),np.round(pearson_p,5)))

    ax[4,1].set_xlim(2500,14000)

    fig.tight_layout()
    plt.savefig(str(savepath+'correlations.png',dpi=300)
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

    fig, ax = plt.subplots(2,3, figsize=(23,15))
    X1 = DZ.Date[0:30]
    X2 = DZ.Date[30:60]
    X3 = DZ.Date[60:90]
    X4 = DZ.Date[90:120]   
    colours = ['b','g','k','y']
    
    ########
    # Albedo
    ########

    Y1 = DZ.meanAlbedo[0:30]
    Y2 = DZ.meanAlbedo[30:60]
    Y3 = DZ.meanAlbedo[60:90]
    Y4 = DZ.meanAlbedo[90:120]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    years = ['2016','2017','2018','2019']
    
    for i in range(len(Xs)):
        X = Xs[i]
        Y = Ys[i]
        Xnums = np.arange(0,len(X),1)
        ax[0,0].plot(Xnums,Y.rolling(5).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnum, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnum, Y)
        ax[0,0].plot(intercept + slope * Xnum, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[0,0].set_xlim(0,33)
    ax[0,0].set_xticks([],[])
    ax[0,0].set_ylabel('Albedo')
    ax[0,0].legend(bbox_to_anchor=(0.8, 1.2))


    ########
    # Algae
    ########

    Y1 = DZ.meanAlgae[0:30]
    Y2 = DZ.meanAlgae[30:60]
    Y3 = DZ.meanAlgae[60:90]
    Y4 = DZ.meanAlgae[90:120]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i]
        Y = Ys[i]
        Xnums = np.arange(0,len(X),1)
        ax[0,1].plot(Xnums,Y.rolling(5).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnum, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnum, Y)
        ax[0,1].plot(intercept + slope * Xnum, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[0,1].set_xlim(0,33)
    ax[0,1].set_xticks([],[])
    ax[0,1].set_ylabel('Algae (ppb)')
    ax[0,1].legend(bbox_to_anchor=(0.80, 1.2))
    

    ############
    # Grain Size
    ############

    Y1 = DZ.meanGrain[0:30]
    Y2 = DZ.meanGrain[30:60]
    Y3 = DZ.meanGrain[60:90]
    Y4 = DZ.meanGrain[90:120]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i]
        Y = Ys[i]
        Xnums = np.arange(0,len(X),1)
        ax[0,2].plot(Xnums,Y.rolling(5).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnum, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnum, Y)
        ax[0,2].plot(intercept + slope * Xnum, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[0,2].set_xlim(0,33)
    ax[0,2].set_xticks([],[])
    ax[0,2].set_ylabel('Grain Size (microns)')
    ax[0,2].legend(bbox_to_anchor=(0.80, 1.2))

    ##########
    # Dust
    ##########

    Y1 = DZ.meanDust[0:30]
    Y2 = DZ.meanDust[30:60]
    Y3 = DZ.meanDust[60:90]
    Y4 = DZ.meanDust[90:120]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i]
        Y = Ys[i]
        Xnums = np.arange(0,len(X),1)
        ax[1,0].plot(Xnums,Y.rolling(5).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnum, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnum, Y)
        ax[1,0].plot(intercept + slope * Xnum, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[1,0].set_xlim(0,33)
    ax[1,0].set_xticks([],[])
    ax[1,0].set_ylabel('Dust (ppb)')
    ax[1,0].legend(bbox_to_anchor=(0.80, 1.2))


    ##########
    # Density
    ##########

    Y1 = DZ.meanDensity[0:30]
    Y2 = DZ.meanDensity[30:60]
    Y3 = DZ.meanDensity[60:90]
    Y4 = DZ.meanDensity[90:120]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i]
        Y = Ys[i]
        Xnums = np.arange(0,len(X),1)
        ax[1,1].plot(Xnums,Y.rolling(5).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnum, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnum, Y)
        ax[1,1].plot(intercept + slope * Xnum, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[1,1].set_xlim(0,33)
    ax[1,1].set_xticks([],[])
    ax[1,1].set_ylabel('Density (kg m-3)')
    ax[1,1].legend(bbox_to_anchor=(0.80, 1.2))

    # Add correlation heatmap for DZ to final panel
    # (this shows correlation between each variable in DZ)

    DZCorr = pd.DataFrame(columns = ['Albedo','Algae','Dust','Grain Size', 'Density','DOY'])
    DZCorr['Albedo'] = DZ.meanAlbedo
    DZCorr['Algae'] = DZ.meanAlgae
    DZCorr['Dust'] = DZ.meanDust
    DZCorr['Grain Size'] = DZ.meanGrain
    DZCorr['Density'] = DZ.meanDensity
    DZCorr['DOY'] = DZ.DOY
    
    ax[1,2] = sns.heatmap(DZCorr.corr(), xticklabels=DZCorr.columns,\
         yticklabels=DZCorr.columns, annot=True, cmap = 'coolwarm')


    plt.savefig(str(savepath+'/JJA_trends.png'),dpi=300)

    return


def Latitudinal_trends(wea,web,wec,wev,weu,wet, savepath):
    
    """
    ADD SOMETHING TO SEE IF THERE IS ANY SYSTEMATIC DIFFERENCE BTEWEEN THE 
    INDIVIDUAL TILES - ANY N-S TRENDS?
    
    """

    return

# Function Calls

filepath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/'
savepath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Figures_and_Tables/'

DZ, wea, web, wec, wet, weu, wev = import_csvs(DZ = True, Tiles=True,\
     filepath)'
AllTileVars(DZ, wea, web, wec, wet, weu, wev, savepath)
DZvars_with_errorbars(DZ, savepath)
correlate_vars(DZ)
annual_trends(DZ)
Latitudinal_trends(wea,web,wec,wev,weu,wet, savepath)

