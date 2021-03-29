"""
This script compares takes field-measured spectra where metadata is available and subtracts
the reflectance at 9 key wavelengths to the same wavelengths in a LUT of DISORT 
simulated spectra. The combination with the lowest absolute error is selected and the 
parameters used in SNICAR to generate the best-matching spectra are added to a list.

This enables manual validation of the LUT-comparison method.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def BDA2_of_field_samples():

    """
    2DBA index calculated from field samples after field spectra are averaged over S2 band 4 and 5 wavelengths
    weighted by the sensor spectral response function for each band. The index is then calculated as B5/B4
    and the cell concentration predicted using Wang et al's (2018) conversion equation.

    """

    spectra = pd.read_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/Training_Data/HCRF_master_16171819.csv')
    
    # reformat LUT: flatten LUT from 3D to 2D array with one column per combination
    # of RT params, one row per wavelength
    
    responsefunc = pd.read_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/S2SpectralResponse.csv')
    func04 = responsefunc['B4'].loc[(responsefunc['SR_WL']>650)&(responsefunc['SR_WL']<=680)]
    func05 = responsefunc['B5'].loc[(responsefunc['SR_WL']>698)&(responsefunc['SR_WL']<=713)]

    filenames = []
    Idx2DBAList =[]
    prd2DBAList = []
    Idx2DBA_S2List =[]
    prd2DBA_S2List = []
    Idx2DBA_Ideal_List = []
    prd2DBA_Ideal_List = []

    for i in np.arange(0,len(spectra.columns),1):
        
        if i != 'Wavelength':

            colname = spectra.columns[i]
            spectrum = np.array(spectra[colname])
            
            B04 = np.mean(spectrum[300:330] * func04)
            B05 = np.mean(spectrum[348:363] * func05)
            Idx2DBA = spectrum[355]/spectrum[315]
            prd2DBA = 10E-35 * Idx2DBA * np.exp(87.015*Idx2DBA)
            Idx2DBA_S2 = B05/B04
            prd2DBA_S2 = 10E-35 * Idx2DBA_S2 * np.exp(87.015*Idx2DBA_S2)
            Idx2DBA_Ideal = spectrum[360]/spectrum[330]
            prd2DBA_Ideal = 10E-35 * Idx2DBA_Ideal * np.exp(87.015*Idx2DBA_Ideal)
            
            filenames.append(colname)
            Idx2DBAList.append(Idx2DBA)
            prd2DBAList.append(prd2DBA)
            Idx2DBA_S2List.append(Idx2DBA_S2)
            prd2DBA_S2List.append(prd2DBA_S2)
            Idx2DBA_Ideal_List.append(Idx2DBA_Ideal)
            prd2DBA_Ideal_List.append(prd2DBA_Ideal)

    Out = pd.DataFrame()
    Out['filename'] = filenames
    Out['2DBAIdx'] = Idx2DBAList
    Out['2DBAPrediction'] = prd2DBAList
    Out['2DBA_S2Idx'] = Idx2DBA_S2List
    Out['2DBA_S2Prediction'] = prd2DBA_S2List
    Out['2DBAIdx_Ideal'] = Idx2DBA_Ideal_List
    Out['2DBAPrediction_Ideal'] = prd2DBA_Ideal_List

    return Out


def compare_predicted_and_measured(savepath):

    ## imports and data organisation
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd


    DF = pd.read_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/RISA_OUT/Spectra_Metadata.csv')

    measured_cells = DF['measured_cells']
    modelled_cells = DF['algae_cells_inv_model_S2']
    BDA2_cells = DF['cells_BDA2_centre_wang']
    BDA2_idx = DF['BDA2_centre']

    
    ## regression models

    # Ordinary least squares regression
    model1 = sm.OLS(modelled_cells,measured_cells).fit()
    summary1 = model1.summary()
    test_x = [0,1000,5000,7500,10000,12500,15000,17500,20000,25000, 30000, 35000, 40000, 50000]
    ypred1 = model1.predict(test_x)

    # regress measured cells against band index 
    # use this to give predictive linear model 

    BDA2_PredModel = sm.OLS(measured_cells,sm.add_constant(BDA2_idx)).fit()
    BDA2_PredModel_r2 = np.round(BDA2_PredModel.rsquared,3)
    BDA2_PredModel_y = BDA2_PredModel.predict(sm.add_constant(BDA2_idx)) 
    
    # regress BDA2 predicted cells against measured cells
    model2 = sm.OLS(BDA2_PredModel_y,measured_cells).fit()
    summary2 = model2.summary()
    test_x = [0,1000,5000,7500,10000,12500,15000,17500,20000,25000, 30000, 35000, 40000, 50000]
    ypred2 = model2.predict(test_x)

    # multipanel figure
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,8))

    ax1.plot(measured_cells,color='k',marker = 'x',\
        label='field-measured')
    ax1.plot(modelled_cells,color='b',marker = 'o', \
        markerfacecolor='None', alpha=0.6, linestyle = 'dashed',\
            label='RTM model prediction')
    ax1.plot(BDA2_PredModel_y,color='r', marker ='^', \
        markerfacecolor='None', alpha=0.6, linestyle = 'dotted',\
            label='new 2BDA model prediction')
    ax1.set_ylabel('Algal concentration (cells/mL)')    
    ax1.set_xticks(range(len(measured_cells)))
    ax1.set_xticklabels([])
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Individual samples')
    ax1.set_ylim(0,65000)

    ax2.scatter(measured_cells, modelled_cells, marker='o',\
        facecolor ='None',color='b',\
        label='RTM\nr$^2$ = {}\np = {}'.format(np.round(model1.rsquared,3),\
            np.round(model1.pvalues[0],8)))
    ax2.plot(test_x, ypred1, linestyle='dotted',color='b')
    ax2.scatter(measured_cells, BDA2_PredModel_y, marker = '^',\
         facecolor='None',color='r', label='2BDA\nr$^2$ = {}\np = {}'.format(\
             np.round(model2.rsquared,3),\
        np.round(model2.pvalues[0],10)))
    ax2.plot(test_x, ypred2, linestyle = 'dashed', color='r')
    ax2.set_ylabel('Algal concentration,\n cells/mL (field)')
    ax2.set_xlabel('Algal concentration,\n clls/mL (predicted by model)')
    ax2.set_xlim(0,50000),ax2.set_ylim(0,60000)

    ax2.legend(loc='upper left')

    fig.tight_layout()

    savepath = '/home/joe/Code/Remote_Ice_Surface_Analyser/Manuscript/Figures'
    fig.savefig(str(savepath+'/measured_modelled_algae.png'),dpi=300)

    return



savepath = '/home/joe/Code/Remote_Ice_Surface_Analyser/RISA_OUT/Figures_and_Tables/'
compare_predicted_and_measured(savepath)
# clean_ice_field_vs_DISORT_NIR()