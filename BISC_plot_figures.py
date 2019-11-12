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
plt.style.use('tableau-colorblind10')

# PLOTTING (if toggled)

###### user-definitions #######

tile_selection='INDIVIDUAL_DATE' # choose "INDIVIDUAL_DATE" or "ALL DATES"
parameters = ['classified','albedo'] # choose one or multiple from ['classified','albedo','grain_size','density','algae','dust']
output_format = 'jpg' # file extension without leading stop (.)
output_res = 100
figsize=(15,15)

# set colormap
cmap1 = mpl.colors.ListedColormap(
    ['purple', 'white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
cmap1.set_under(color='white')  # make sure background is white
cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
cmap2.set_under(color='white')  # make sure background is white

# file selection from output dir
if tile_selection=='INDIVIDUAL_DATE':
    
    tileID = '22wev'
    path = '/home/joe/Code/BigIceSurfClassifier/Process_Dir/outputs/'
    date = '20160601'

    filelist = glob.glob(str(path + tileID + "/" + tileID + "_" + date + "*Classification_and_Albedo_Data.nc"))

elif tile_selection=='ALLDATES':

    filelist = glob.glob(str(path + tileID + '/' + '*Classification_and_Albedo_Data.nc'))


# looping through files and params
for filename in filelist:
    
    start = str(path+tileID+'/'+tileID+'_')
    end = "_Classification_and_Albedo_Data.nc"
    date = filename[len(start):-len(end)]

    for param in parameters:
        
        if param=='classified':
            tile = xr.open_dataset(filename)
            plt.figure(figsize=figsize)
            plt.imshow(tile.classified.values, cmap=cmap1, vmin=0, vmax=6),plt.colorbar()
            plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
            # plt.title('Greenland Ice Sheet from Sentinel 2 classified using Random Forest Classifier (top) and albedo (bottom)')
            plt.rcParams["axes.grid"] = False
            plt.savefig(str(path + f'{param}_{tileID}_{date}.{output_format}'),facecolor='w',dpi=output_res)

        elif param=='albedo':
            tile = xr.open_dataset(filename)
            plt.figure(figsize=figsize)
            plt.imshow(tile.albedo.values, cmap=cmap2, vmin=0, vmax=1),plt.colorbar()
            plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
            plt.rcParams["axes.grid"] = False
            plt.savefig(str(path + f'{param}_{tileID}_{date}.{output_format}'),facecolor='w',dpi=output_res)
        

        ### continue elifs for other params...