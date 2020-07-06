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




def import_csvs(filepath, DZ = True, Tiles = True, byClass = True):
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
            DZ_Class.Date = pd.to_datetime(DZ_Class.Date)
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

    return DZ, DZ_Class, wea, web, wec, wet, weu, wev



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
    plt.savefig(str(savepath+'correlations.png',dpi=300))
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
    X1 = DZ.loc[DZ['Date'].dt.year==2016]
    X2 = DZ.loc[DZ['Date'].dt.year==2017]
    X3 = DZ.loc[DZ['Date'].dt.year==2018]
    X4 = DZ.loc[DZ['Date'].dt.year==2019]
    XX1 = DZ_Class.loc[DZ_Class['Date'].dt.year==2016]
    XX2 = DZ_Class.loc[DZ_Class['Date'].dt.year==2017]
    XX3 = DZ_Class.loc[DZ_Class['Date'].dt.year==2018]
    XX4 = DZ_Class.loc[DZ_Class['Date'].dt.year==2019]   
    colours = ['b','g','k','y']
    
    ########
    # Albedo
    ########

    Y1 = DZ['meanAlbedo'].loc[DZ['Date'].dt.year==2016]
    Y2 = DZ['meanAlbedo'].loc[DZ['Date'].dt.year==2017]
    Y3 = DZ['meanAlbedo'].loc[DZ['Date'].dt.year==2018]
    Y4 = DZ['meanAlbedo'].loc[DZ['Date'].dt.year==2019]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    years = ['2016','2017','2018','2019']
    
    for i in range(len(Xs)):
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[0,0].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[0,0].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[0,0].set_xlim(0,26)
    ax[0,0].set_xticks([],[])
    ax[0,0].set_ylabel('Albedo')
    ax[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[0,0].text(0,0.81,'Albedo',fontsize=22)

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
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[0,1].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[0,1].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[0,1].set_xlim(0,26)
    ax[0,1].set_xticks([],[])
    ax[0,1].set_ylabel('Algae (ppb)')
    ax[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[0,1].text(0,9300,'Algae (ppb)',fontsize=22)

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
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[0,2].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[0,2].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[0,2].set_xlim(0,26)
    ax[0,2].set_xticks([],[])
    ax[0,2].set_ylabel('Grain Size (microns)')
    ax[0,2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[0,2].text(0,13500,'Grain Size (microns)',fontsize=22)
    
    ##########
    # Dust
    ##########

    Y1 = DZ['meanDust'].loc[DZ['Date'].dt.year==2016]
    Y2 = DZ['meanDust'].loc[DZ['Date'].dt.year==2017]
    Y3 = DZ['meanDust'].loc[DZ['Date'].dt.year==2018]
    Y4 = DZ['meanDust'].loc[DZ['Date'].dt.year==2019]
    Xs = [X1,X2,X3,X4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[1,0].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums,Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[1,0].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[1,0].set_xlim(0,26)
    ax[1,0].set_xticks([],[])
    ax[1,0].set_ylabel('Dust (ppb)')
    ax[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[1,0].text(0,36900,'Dust (ppb)',fontsize=22)

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
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[1,1].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[1,1].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[1,1].set_xlim(0,26)
    ax[1,1].set_xticks([],[])
    ax[1,1].set_ylabel('Density (kg m-3)')
    ax[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[1,1].text(0,820,'Density (kgm-3)',fontsize=22)

    ###############
    # SNOW COVERAGE
    ###############

    Y1 = DZ_Class['ClassCountSN'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['ClassCountSN'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['ClassCountSN'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['ClassCountSN'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[1,2].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[1,2].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[1,2].set_xlim(0,26)
    ax[1,2].set_xticks([],[])
    ax[1,2].set_ylabel('N pixels (SNOW) x 1e7')
    ax[1,2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[1,2].text(0,1.36e7,'N Pixels: Snow',fontsize=22)

    ###############
    # CI COVERAGE
    ###############

    Y1 = DZ_Class['ClassCountCI'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['ClassCountCI'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['ClassCountCI'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['ClassCountCI'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[2,0].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[2,0].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[2,0].set_xlim(0,26)
    ax[2,0].set_xticks([],[])
    ax[2,0].set_ylabel('N pixels (CLEAN ICE) x 1e7')
    ax[2,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[2,0].text(0,1.18e7,'N pixels: Clean Ice',fontsize=22)

    ###############
    # HA COVERAGE
    ###############

    Y1 = DZ_Class['ClassCountHA'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['ClassCountHA'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['ClassCountHA'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['ClassCountHA'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[2,1].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)
        ax[2,1].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))
    ax[2,1].set_xlim(0,26)
    ax[2,1].set_xticks([],[])
    ax[2,1].set_ylabel('N pixels (HEAVY ALGAE)')
    ax[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[2,1].text(0,1500000,'N pixels: Heavy Algae',fontsize=22)

    ###############
    # LA COVERAGE
    ###############

    Y1 = DZ_Class['ClassCountLA'].loc[DZ_Class['Date'].dt.year==2016]
    Y2 = DZ_Class['ClassCountLA'].loc[DZ_Class['Date'].dt.year==2017]
    Y3 = DZ_Class['ClassCountLA'].loc[DZ_Class['Date'].dt.year==2018]
    Y4 = DZ_Class['ClassCountLA'].loc[DZ_Class['Date'].dt.year==2019]
    Xs = [XX1,XX2,XX3,XX4]
    Ys = [Y1,Y2,Y3,Y4]
    
    for i in range(len(Xs)):
        
        X = Xs[i][0:26]
        Y = Ys[i][0:26]
        Xnums = np.arange(0,len(X),1)
        ax[2,2].plot(Xnums,Y.rolling(3).mean(),color=colours[i]) 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xnums, Y)
        pearson_coef, pearson_p = stats.pearsonr(Xnums, Y)

        ax[2,2].plot(intercept + slope * Xnums, color=colours[i], linestyle='-',\
             linewidth=0.5, label = '{}: r2 = {}, R = {}, p = {}'.format(years[i],np.round(r_value**2,2),\
                 np.round(pearson_coef,2),np.round(p_value,3)))

    ax[2,2].set_xlim(0,26)
    ax[2,2].set_xticks([],[])
    ax[2,2].set_ylabel('N pixels (LIGHT ALGAE)')
    ax[2,2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=1)
    ax[2,2].text(0,6700000,'N pixels: Light Algae',fontsize=22)
    
    fig.tight_layout()

    fig.savefig(str(savepath+'/JJA_trends.png'),dpi=300)

    return


def Latitudinal_trends(wea,web,wec,wev,weu,wet, savepath,\
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

        kruskalDF =pd.DataFrame(columns = ['wea','web','wec','wet','weu','wev'])
        kruskalDF['wea'] = wea.meanDust
        kruskalDF['web'] = web.meanDust
        kruskalDF['wec'] = wec.meanDust
        kruskalDF['wet'] = wet.meanDust
        kruskalDF['weu'] = weu.meanDust
        kruskalDF['wev'] = wev.meanDust

        Col1 = kruskalDF['wea']
        Col2 = kruskalDF['web']
        Col3 = kruskalDF['wec']
        Col4 = kruskalDF['wet']
        Col5 = kruskalDF['weu']
        Col6 = kruskalDF['wev']

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
        KruskalWallisTest()

    
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

    DZCorr = DZ[['meanAlbedo','meanGrain','meanDensity','meanAlgae','meanDust','DOY']]
    DZCorr.columns = [['Albedo','Grain Size','Density','Algae','Dust','DOY']]
    
    DZ_Class_Corr = DZ_Class[['ClassCountSN','ClassCountCI','ClassCountHA','ClassCountLA','ClassCountWAT','ClassCountCC']]
    DZ_Class_Corr.columns=[['Snow','Clean Ice','Heavy Algae','Light Algae','Water','Cryoconite']]
    DZ_Class_Corr['TotalAlg'] = DZ_Class[['ClassCountHA','ClassCountLA']].sum(axis=1)

    fig, ax = plt.subplots(1,2,figsize=(20,8))
    ax[0] = sns.heatmap(DZCorr.corr(),annot=True,cmap='coolwarm',ax=ax[0])
    ax[0].set_title('RT variables')
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)

    ax[1] = sns.heatmap(DZ_Class_Corr.corr(),annot=True,cmap='coolwarm',ax=ax[1])
    ax[1].set_title('Spatial Coverage')
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    fig.savefig(str(savepath+'Heatmaps.png'),dpi=300)

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


# Function Calls

filepath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/'
savepath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Figures_and_Tables/'

DZ, DZ_Class, wea, web, wec, wet, weu, wev = import_csvs(filepath,True,True)
# AllTileVars(DZ, wea, web, wec, wet, weu, wev, savepath)
# DZvars_with_errorbars(DZ, savepath)
# correlate_vars(DZ)
# corr_heatmaps(DZ,DZ_Class,savepath)
# annual_trends(DZ,savepath)
# Latitudinal_trends(wea,web,wec,wev,weu,wet, savepath,\
# anova_assumptions = True, KW_test = True)
# plot_coverage(DZ, DZ_Class,savepath)

