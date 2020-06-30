import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import os
import pandas as pd
import datetime

OutDataPath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Raw/*[0-9].nc'
ClassDataPath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Raw/*byClass.nc'


def process_from_nc(OutDataPath, ClassDataPath, byClass=False):
    # set paths for grabbing correct files - either by class or totals
    
    if not byClass:

        DataPath = OutDataPath

        DataList = glob.glob(DataPath)
        
        for i in DataList:
            
            # grab tile and year from filename
            tile = i[53:58]
            year = i[59:63]

            # open file using xarray and convert to multi-indexed dataframe
            file = xr.open_dataarray(i)
            DF = file.to_dataframe(name='DF')

            # create empty dataframe and numpy array ready to receive output data
            OutDF = pd.DataFrame(columns=['Date','STDAlbedo','STDAlgae','STDDensity','STDDust',\
            'STDGrain','meanAlbedo','meanAlgae','meanDensity','meanDust','meanGrain'])
            
            OutArray = np.zeros(shape=(len(DF.index.levels[1]),len(DF.index.levels[0])))
            dateList=[]

            # grab list of dates from index level 0
            # grab list of variables from index level 1
            dates = DF.index.levels[1]
            variables = DF.index.levels[0]
            varcounter = 0
            
            for i in variables:
                tempList =[]

                for j in dates:
                                
                    temp = DF.loc[(i,j)].values
                
                    tempList.append(temp)

                OutArray[:,varcounter] = tempList
                OutDF[i] = OutArray[:,varcounter]
                varcounter +=1
            
            OutDF['Date'] = dates
            OutPath = str('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/' + tile + "_" + year + "_" + 'processed.csv')
            OutDF.to_csv(OutPath,index=False)

    else: 
        # REQUIRES CODE FOR PROCESSING CLASS DATA HERE!!!!!
        
        pass

    
    
    print("NETCDFs converted successfully \n.csvs now available in PROCESSED folder")

    return


def combineCSVs_byTile():

    for tile in ['22wea','22web','22wec','22wet','22weu','22wev']:

        files = glob.glob(str('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/' + tile + '*'))

        tempList = []

        for file in files:
            df = pd.read_csv(file, index_col=None, header=0)
            tempList.append(df)

        frame = pd.concat(tempList, axis=0, ignore_index=True)
        frame.to_csv(str('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/'+tile+'.csv'),index=False)

    return 


def add_DOYcolumn():

    weaDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wea.csv',index_col=False)
    weaDF = weaDF.sort_values(by='Date').reset_index(drop=True)
    weaDF['Date'] = pd.to_datetime(weaDF['Date'].astype(str), format='%Y/%m/%d')
    weaDF['DOY'] = weaDF['Date'].dt.dayofyear
    weaDF.to_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wea.csv')

    webDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22web.csv',index_col=False)
    webDF = webDF.sort_values(by='Date').reset_index(drop=True)
    webDF['Date'] = pd.to_datetime(webDF['Date'].astype(str), format='%Y/%m/%d')
    webDF['DOY'] = webDF['Date'].dt.dayofyear
    webDF.to_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22web.csv')
    
    wecDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wec.csv',index_col=False)
    wecDF = wecDF.sort_values(by='Date').reset_index(drop=True)
    wecDF['Date'] = pd.to_datetime(wecDF['Date'].astype(str), format='%Y/%m/%d')
    wecDF['DOY'] = wecDF['Date'].dt.dayofyear
    wecDF.to_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wec.csv')

    wetDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wet.csv',index_col=False)
    wetDF = wetDF.sort_values(by='Date').reset_index(drop=True)
    wetDF['Date'] = pd.to_datetime(wetDF['Date'].astype(str), format='%Y/%m/%d')
    wetDF['DOY'] = wetDF['Date'].dt.dayofyear
    wetDF.to_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wet.csv')
    
    weuDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22weu.csv',index_col=False)
    weuDF = weuDF.sort_values(by='Date').reset_index(drop=True)
    weuDF['Date'] = pd.to_datetime(weuDF['Date'].astype(str), format='%Y/%m/%d')
    weuDF['DOY'] = weuDF['Date'].dt.dayofyear
    weuDF = weuDF.interpolate(kind='cubic')
    weuDF.to_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22weu.csv')

    wevDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wev.csv',index_col=False)
    wevDF = wevDF.sort_values(by='Date').reset_index(drop=True)
    wevDF['Date'] = pd.to_datetime(wevDF['Date'].astype(str), format='%Y/%m/%d')
    wevDF['DOY'] = wevDF['Date'].dt.dayofyear
    wevDF.to_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wev.csv')
    
    return

def combine_darkzone():

    weaDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wea.csv')
    webDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22web.csv')
    wecDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wec.csv')
    wetDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wet.csv')
    weuDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22weu.csv')
    wevDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wev.csv')

    meanDF = pd.DataFrame(columns=['Date','STDAlbedo','STDAlgae','STDDust','STDGrain',\
        'meanAlbedo','meanAlgae','meanDensity','meanDust','meanGrain'])

    meanDF['Date'] = pd.to_datetime(weaDF['Date'].astype(str), format='%Y/%m/%d')
    meanDF['DOY'] = meanDF['Date'].dt.dayofyear
    meanDF['Date'] = pd.to_datetime(meanDF['Date'], format='%Y%m%d')

    meanDF['STDAlbedo'] = np.mean([weaDF['STDAlbedo'],webDF['STDAlbedo'],wecDF['STDAlbedo'],\
        wetDF['STDAlbedo'],weuDF['STDAlbedo'],wevDF['STDAlbedo']],axis=0)
    
    meanDF['STDAlgae'] = np.mean([weaDF['STDAlgae'],webDF['STDAlgae'],wecDF['STDAlgae'],\
        wetDF['STDAlgae'],wevDF['STDAlgae']],axis=0)

    meanDF['STDDensity'] = np.mean([weaDF['STDDensity'],webDF['STDDensity'],wecDF['STDDensity'],\
        wetDF['STDDensity'],wevDF['STDDensity']],axis=0)

    meanDF['STDDust'] = np.mean([weaDF['STDDust'],webDF['STDDust'],wecDF['STDDust'],\
        wetDF['STDDust'],wevDF['STDDust']],axis=0)

    meanDF['STDGrain'] = np.mean([weaDF['STDGrain'],webDF['STDGrain'],wecDF['STDGrain'],\
        wetDF['STDGrain'],wevDF['STDGrain']],axis=0)

    meanDF['meanAlbedo'] = np.mean([weaDF['meanAlbedo'],webDF['meanAlbedo'],wecDF['meanAlbedo'],\
        wetDF['meanAlbedo'],wevDF['meanAlbedo']],axis=0)

    meanDF['meanAlgae'] = np.mean([weaDF['meanAlgae'],webDF['meanAlgae'],wecDF['meanAlgae'],\
        wetDF['meanAlgae'],wevDF['meanAlgae']],axis=0)

    meanDF['meanDensity'] = np.mean([weaDF['meanDensity'],webDF['meanDensity'],wecDF['meanDensity'],\
        wetDF['meanDensity'],wevDF['meanDensity']],axis=0)

    meanDF['meanDust'] = np.mean([weaDF['meanDust'],webDF['meanDust'],wecDF['meanDust'],\
        wetDF['meanDust'],wevDF['meanDust']],axis=0)

    meanDF['meanGrain'] = np.mean([weaDF['meanGrain'],webDF['meanGrain'],wecDF['meanGrain'],\
        wetDF['meanGrain'],wevDF['meanGrain']],axis=0)

    meanDF = meanDF.sort_values(by='Date').reset_index(drop=True)
    meanDF.to_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/WholeDZ.csv',index=False)

    return meanDF


def clear_temp_files(clearOutData= False,clearClassData = False, clearCSVs = False):
    
    clearOutData=True

    if clearOutData:
        files = glob.glob('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Raw/*[0-9].nc')
        for file in files:
            os.remove(file)

    if clearClassData:
        files = glob.glob('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Raw/*byClass.nc')
        for file in files:
            os.remove(file)

    if clearCSVs:  
        files = glob.glob('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/*processed.csv')
        for file in files:
            os.remove(file)
    
    return


process_from_nc(OutDataPath, ClassDataPath, byClass = False)
combineCSVs_byTile()
add_DOYcolumn()
meanDF = combine_darkzone()
#clear_temp_files(clearOutData= True, clearClassData = False, clearCSVs = True)

