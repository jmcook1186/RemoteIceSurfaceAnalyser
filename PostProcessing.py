import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob as glob
import pandas as pd

OutDataPath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/*[0-9].nc'
ClassDataPath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/*byClass.nc'

def process_from_nc(OutDataPath, ClassDataPath, byClass=False):
    # set paths for grabbing correct files - either by class or totals
    
    if byClass:
        DataPath = ClassDataPath
    else:
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

        if byClass:

            OutPath = str('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/' + tile + "_" + year + "_" + 'processed_byCLASS.csv') 
        
        else:
            OutPath = str('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/' + tile + "_" + year + "_" + 'processed.csv')

        OutDF.to_csv(OutPath,index=False)

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


def combine_darkzone():

    weaDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wea.csv')
    webDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22web.csv')
    wecDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wec.csv')
    wetDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wet.csv')
    weuDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22weu.csv')
    wevDF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/22wev.csv')

    meanDF = pd.DataFrame(columns=['Date','STDAlbedo','STDAlgae','STDDust','STDGrain',\
        'meanAlbedo','meanAlgae','meanDensity','meanDust','meanGrain'])

    meanDF['Date'] = weaDF['Date']
    meanDF['Date'] = pd.to_datetime(meanDF['Date'], format='%Y%m%d')

    meanDF = meanDF.sort_values(by='Date').reset_index(drop=True)

    meanDF['STDAlbedo'] = np.mean([weaDF['STDAlbedo'],webDF['STDAlbedo'],wecDF['STDAlbedo'],\
        wetDF['STDAlbedo'],weuDF['STDAlbedo'],wevDF['STDAlbedo']])
    
    meanDF['STDAlgae'] = np.mean([weaDF['STDAlgae'],webDF['STDAlgae'],wecDF['STDAlgae'],\
        wetDF['STDAlgae'],weuDF['STDAlgae'],wevDF['STDAlgae']])

    meanDF['STDDensity'] = np.mean([weaDF['STDDensity'],webDF['STDDensity'],wecDF['STDDensity'],\
        wetDF['STDDensity'],weuDF['STDDensity'],wevDF['STDDensity']])

    meanDF['STDDust'] = np.mean([weaDF['STDDust'],webDF['STDDust'],wecDF['STDDust'],\
        wetDF['STDDust'],weuDF['STDDust'],wevDF['STDDust']])

    meanDF['STDGrain'] = np.mean([weaDF['STDGrain'],webDF['STDGrain'],wecDF['STDGrain'],\
        wetDF['STDGrain'],weuDF['STDGrain'],wevDF['STDGrain']])

    meanDF['meanAlbedo'] = np.mean([weaDF['meanAlbedo'],webDF['meanAlbedo'],wecDF['meanAlbedo'],\
        wetDF['meanAlbedo'],weuDF['meanAlbedo'],wevDF['meanAlbedo']])

    meanDF['meanAlgae'] = np.mean([weaDF['meanAlgae'],webDF['meanAlgae'],wecDF['meanAlgae'],\
        wetDF['meanAlgae'],weuDF['meanAlgae'],wevDF['meanAlgae']])

    meanDF['meanDensity'] = np.mean([weaDF['meanDensity'],webDF['meanDensity'],wecDF['meanDensity'],\
        wetDF['meanDensity'],weuDF['meanDensity'],wevDF['meanDensity']])

    meanDF['meanDust'] = np.mean([weaDF['meanDust'],webDF['meanDust'],wecDF['meanDust'],\
        wetDF['meanDust'],weuDF['meanDust'],wevDF['meanDust']])

    meanDF['meanGrain'] = np.mean([weaDF['meanGrain'],webDF['meanGrain'],wecDF['meanGrain'],\
        wetDF['meanGrain'],weuDF['meanGrain'],wevDF['meanGrain']])

    meanDF.to_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/PROCESSED/WholeDZ.csv')

    return meanDF

process_from_nc(DataPath, ClassDataPath, byClass = False)
combineCSVs_byTile()
meanDF = combine_darkzone()