from Plotting_and_Analysis_Functions import *
import matplotlib.pyplot as plt
import numpy as np

## SET VARIABLES
savepath = '/datadrive2/BigIceSurfClassifier/Process_Dir/outputs/'
var = 'reff'
vmin = 500
vmax = 1000
cmap = 'cividis' #'Greys_r' 'BuPu' YlOrBr' 'YlGn'
corr_var1 = 'reff'
corr_var2 = 'density'
dpi = 300


# RUN FUNCTIONS

#CREATE DATASETS (genrates annual and monthly mean datasets: do once then files are available)
# for tile in ["22WEA","22WEB","22WEC","22WET","22WEU","22WEV"]:
#     for year in ["2016","2017","2018","2019"]:
#         create_annual_mean_datasets(year,tile,savepath)
#         create_monthly_mean_datasets(year,tile,savepath)


#for year in ["2016","2017","2018","2019"]:
    #JJA_stats(savepath,var,year)
    #annual_histograms(savepath,var,year,dpi=dpi)
    #annual_maps(savepath, var=var, year=year, vmin=vmin, vmax=vmax, cmap=cmap, dpi=dpi)
    #create_monthly_maps(savepath, var=var, year=year, vmin=vmin, vmax=vmax, cmap =cmap, dpi=dpi)
    #time_series(savepath,var)
    #correlations_btwn_vars(corr_var1, corr_var2, year, savepath)
    #annual_stats(savepath,var,year)
corr_heatmaps(savepath)