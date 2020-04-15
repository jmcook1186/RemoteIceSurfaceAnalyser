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

#############################################################
#############################################################
# USER DEFINITIONS: SET VALUES THEN RUN SCRIPT

tile_selection='ALLDATES' # choose "INDIVIDUAL_DATE" or "ALL DATES"
tileID = '22wev'
year = '2016'
path = str('/datadrive/BigIceSurfClassifier/Process_Dir/outputs/' + tileID + '/')
date = '20160601'
parameters = ['classified','albedo']#,'grain_size','density','algae','dust']
output_format = 'jpg' # file extension without leading stop (.)
output_res = 150
figsize=(15,15)
interactive_plotting=False
savefig=True

file_path = str(path+"FULL_OUTPUT_" + tileID + '_' + year + '.nc')
#############################################################
#############################################################


# toggle interactive plotting (toggle off for batch figure saving)
if interactive_plotting:
    pass
else:
    plt.ioff()

# set colormap
cmap1 = mpl.colors.ListedColormap(
    ['purple', 'white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
cmap1.set_under(color='white')  # make sure background is white

cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
cmap2.set_under(color='white')  # make sure background is white


# file selection from output dir
if tile_selection=='INDIVIDUAL_DATE':
    
    tile = xr.open_dataset(file_path)
    tile = file.sel(date = date)

    for param in parameters:
        
        if param=='classified':
            plt.figure(figsize=figsize)
            plt.imshow(tile.classified.values, cmap=cmap1, vmin=0, vmax=6),plt.colorbar()
            plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
            plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
            plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
            plt.rcParams["axes.grid"] = False
            if savefig:
                plt.savefig(str(path + tileID + f'{param}_{tileID}_{date}.{output_format}'),facecolor='w',dpi=output_res)

        elif param=='albedo':
            plt.figure(figsize=figsize)
            plt.imshow(tile.albedo.values, cmap=cmap2, vmin=0, vmax=1),plt.colorbar()
            plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
            plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
            plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
            plt.rcParams["axes.grid"] = False
            if savefig:
                plt.savefig(str(path + tileID + f'{param}_{tileID}_{date}.{output_format}'),facecolor='w',dpi=output_res)

        elif param=='grain_size':
            plt.figure(figsize=figsize)
            plt.imshow(tile.grain_size.values, cmap='BuPu'),plt.colorbar()
            plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
            plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
            plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
            plt.rcParams["axes.grid"] = False
            if savefig:
                plt.savefig(str(path + tileID + f'{param}_{tileID}_{date}.{output_format}'),facecolor='w',dpi=output_res)
        
        elif param=='density':
            plt.figure(figsize=figsize)
            plt.imshow(tile.density.values, cmap='GnBu'),plt.colorbar()
            plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
            plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
            plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
            plt.rcParams["axes.grid"] = False
            if savefig:
                plt.savefig(str(path + tileID + f'{param}_{tileID}_{date}.{output_format}'),facecolor='w',dpi=output_res)

        elif param=='dust':
            plt.figure(figsize=figsize)
            plt.imshow(tile.dust.values, cmap='cividis'),plt.colorbar()
            plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
            plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
            plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
            plt.rcParams["axes.grid"] = False
            if savefig:
                plt.savefig(str(path + tileID + f'{param}_{tileID}_{date}.{output_format}'),facecolor='w',dpi=output_res)

        elif param=='algae':
            tile = xr.open_dataset(filename)
            plt.figure(figsize=figsize)
            plt.imshow(tile.algae.values, cmap='plasma'),plt.colorbar()
            plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
            plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
            plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
            plt.rcParams["axes.grid"] = False
            if savefig:
                plt.savefig(str(path + tileID + f'{param}_{tileID}_{date}.{output_format}'),facecolor='w',dpi=output_res)
    
        if tile_selection == "ALLDATES":
            if interactive_plotting == False:
                plt.close() # prevent too many open figs in batches by closing plt between dates


elif tile_selection=='ALLDATES':

    tiles = xr.open_dataset(file_path)
    tile_datelist = tiles.date.values

    # looping through files and params
    for tile_date in tile_datelist:
        
        tile = tiles.sel(date=tile_date)
        
        for param in parameters:
            
            if param=='classified':
                plt.figure(figsize=figsize)
                plt.imshow(tile.classified.values, cmap=cmap1, vmin=0, vmax=6),plt.colorbar()
                plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
                plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
                plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
                plt.rcParams["axes.grid"] = False
                if savefig:
                    plt.savefig(str(path + tileID + f'{param}_{tileID}_{tile_date}.{output_format}'),facecolor='w',dpi=output_res)

            elif param=='albedo':
                plt.figure(figsize=figsize)
                plt.imshow(tile.albedo.values, cmap=cmap2, vmin=0, vmax=1),plt.colorbar()
                plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
                plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
                plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
                plt.rcParams["axes.grid"] = False
                if savefig:
                    plt.savefig(str(path + tileID + f'{param}_{tileID}_{tile_date}.{output_format}'),facecolor='w',dpi=output_res)

            elif param=='grain_size':
                plt.figure(figsize=figsize)
                plt.imshow(tile.grain_size.values, cmap='BuPu'),plt.colorbar()
                plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
                plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
                plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
                plt.rcParams["axes.grid"] = False
                if savefig:
                    plt.savefig(str(path + tileID + f'{param}_{tileID}_{tile_date}.{output_format}'),facecolor='w',dpi=output_res)
            
            elif param=='density':
                plt.figure(figsize=figsize)
                plt.imshow(tile.density.values, cmap='GnBu'),plt.colorbar()
                plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
                plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
                plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
                plt.rcParams["axes.grid"] = False
                if savefig:
                    plt.savefig(str(path + tileID + f'{param}_{tileID}_{tile_date}.{output_format}'),facecolor='w',dpi=output_res)

            elif param=='dust':
                plt.figure(figsize=figsize)
                plt.imshow(tile.dust.values, cmap='cividis'),plt.colorbar()
                plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
                plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
                plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
                plt.rcParams["axes.grid"] = False
                if savefig:
                    plt.savefig(str(path + tileID + f'{param}_{tileID}_{tile_date}.{output_format}'),facecolor='w',dpi=output_res)

            elif param=='algae':
                plt.figure(figsize=figsize)
                plt.imshow(tile.algae.values, cmap='plasma'),plt.colorbar()
                plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
                plt.xticks(np.arange(0,len(tile.x),len(tile.x)/2), (tile.x.values[0],tile.x.values[int(len(tile.x)/2)],tile.x.values[-1]),rotation=30)
                plt.yticks(np.arange(0,len(tile.y),len(tile.y)/2), (tile.y.values[0],tile.y.values[int(len(tile.y)/2)],tile.y.values[-1]),rotation=30)
                plt.rcParams["axes.grid"] = False
                if savefig:
                    plt.savefig(str(path + tileID + f'{param}_{tileID}_{tile_date}.{output_format}'),facecolor='w',dpi=output_res)
        
            if interactive_plotting == False:
                plt.close() # prevent too many open figs in batches by closing plt between dates

