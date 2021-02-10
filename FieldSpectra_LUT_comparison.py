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


def grab_params():

    """
    This function takes all field spectra and compares them 
    to the SNICAR simulated spectra LUT used in the RISA project. 

    The resulting predicted params have been collated with the 
    site metadata to be loaded in future instead of generating 
    the data again using this function.

    Results available in: 
    /home/joe/Code/Remote_Ice_Surface_Classifier/RISA_OUT/Spectra_Metadata.csv


    """

    dz = [0.001, 0.01, 0.02, 0.02, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.2]
    densities = [350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    grain_rds = [1000, 2000, 3000, 4000, 5000]
    algae = [0, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 
    200000, 250000, 300000, 350000, 400000]
    zeniths = [37, 45, 53, 60] # = coszen 80, 70, 60, 50
    
    spectra = pd.read_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/Training_Data/HCRF_master_16171819.csv')
    LUT = np.load('/home/joe/Code/BioSNICAR_GO_PY/Spec_LUT_70.npy')
    
    densityList = []
    grainlist = []
    algaeList = []
    filenames = []
    index2DBA = []

    # reformat LUT: flatten LUT from 3D to 2D array with one column per combination
    # of RT params, one row per wavelength

    wavelengths = np.arange(0.2,5,0.01)
    LUT = LUT.reshape(len(densities)*len(grain_rds)*len(algae),len(wavelengths))
    idx = [29, 36, 46, 50, 54, 58, 66, 141, 200]
    LUT = LUT[:,idx] # reduce wavelengths to only the 9 that match the S2 image

    
    for i in np.arange(0,len(spectra.columns),1):
        
        if i != 'wavelength':

            colname = spectra.columns[i]
            spectrum = np.array(spectra[colname][idx])
            error_array = abs(LUT - spectrum)
            mean_error = np.mean(error_array,axis=1)
            index = np.argmin(mean_error)
            param_idx = np.unravel_index(index,[len(densities),len(grain_rds),len(algae)])

            filenames.append(colname)
            densityList.append(densities[param_idx[0]])
            grainlist.append(grain_rds[param_idx[1]])
            algaeList.append(algae[param_idx[2]])
            index2DBA.append(spectrum[3]/spectrum[2])
            

    Out = pd.DataFrame(columns=['filename','density','grain','algae','index2DBA'])
    Out['filename'] = filenames
    Out['density'] = densityList
    Out['grain'] = grainlist
    Out['algae'] = algaeList
    Out['index2DBA'] = index2DBA

    return Out


def band_ratio_heatmaps():
    
    """
    plot heatmap of band ratio value (color) for range of grain size and density

    """

    LUT = np.load('/home/joe/Code/BioSNICAR_GO_PY/Spec_LUT_50.npy')
    
    densities = [350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    grain_rds = [1000, 2000, 3000, 4000, 5000]
    algae = [0, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 
    200000, 250000, 300000, 350000, 400000]

    data = np.zeros(shape=(len(algae),len(densities),len(grain_rds)))

    for i in np.arange(0,len(algae),1):
        for j in np.arange(0,len(densities),1):
            for k in np.arange(0,len(grain_rds),1):            
                
                spectrum = LUT[j,k,i,:]

                band_ratio = spectrum[51]/spectrum[47]
                cells = 10e-35 * band_ratio * np.exp(87.015*band_ratio)
                
                data[i,j,k] = band_ratio


    fig,axes = plt.subplots(2,3,figsize=(8,8))
    
    im1 = axes[0,0].imshow(data[0,:,:])
    im2 = axes[0,1].imshow(data[1,:,:])
    im3 = axes[0,2].imshow(data[2,:,:])
    im4 = axes[1,0].imshow(data[3,:,:])
    im5 = axes[1,1].imshow(data[4,:,:])
    im6 = axes[1,2].imshow(data[5,:,:])

    fig.colorbar(im1, ax=axes[0, 0])
    fig.colorbar(im2, ax=axes[0, 1])
    fig.colorbar(im3, ax=axes[0, 2])
    fig.colorbar(im4, ax=axes[1, 0])
    fig.colorbar(im5, ax=axes[1, 1])
    fig.colorbar(im6, ax=axes[1, 2])

    axes[0,0].set_xticks(np.arange(0,len(grain_rds),1))
    axes[0,0].set_xticklabels(grain_rds,rotation=30)
    axes[0,0].set_yticks(np.arange(0,len(densities),1))
    axes[0,0].set_yticklabels(densities)
    axes[0,0].set_ylabel('density (kg m-3)')
    axes[0,0].set_xlabel('r_eff (microns)')
    axes[0,0].set_title('M_alg: \n{} ppb'.format(algae[0]))

    axes[0,1].set_xticks(np.arange(0,len(grain_rds),1))
    axes[0,1].set_xticklabels(grain_rds,rotation=30)
    axes[0,1].set_yticks(np.arange(0,len(densities),1))
    axes[0,1].set_yticklabels(densities)
    axes[0,1].set_ylabel('density (kg m-3)')
    axes[0,1].set_xlabel('r_eff (microns)')
    axes[0,1].set_title('M_alg: \n{} ppb'.format(algae[1]))

    axes[0,2].set_xticks(np.arange(0,len(grain_rds),1))
    axes[0,2].set_xticklabels(grain_rds,rotation=30)
    axes[0,2].set_yticks(np.arange(0,len(densities),1))
    axes[0,2].set_yticklabels(densities)
    axes[0,2].set_ylabel('density (kg m-3)')
    axes[0,2].set_xlabel('r_eff (microns)')
    axes[0,2].set_title('M_alg: \n{} ppb'.format(algae[2]))

    axes[1,0].set_xticks(np.arange(0,len(grain_rds),1))
    axes[1,0].set_xticklabels(grain_rds,rotation=30)
    axes[1,0].set_yticks(np.arange(0,len(densities),1))
    axes[1,0].set_yticklabels(densities)
    axes[1,0].set_ylabel('density (kg m-3)')
    axes[1,0].set_xlabel('r_eff (microns)')
    axes[1,0].set_title('M_alg: \n{} ppb'.format(algae[3]))

    axes[1,1].set_xticks(np.arange(0,len(grain_rds),1))
    axes[1,1].set_xticklabels(grain_rds,rotation=30)
    axes[1,1].set_yticks(np.arange(0,len(densities),1))
    axes[1,1].set_yticklabels(densities)
    axes[1,1].set_ylabel('density (kg m-3)')
    axes[1,1].set_xlabel('r_eff (microns)')
    axes[1,1].set_title('M_alg: \n{} ppb'.format(algae[4]))

    axes[1,2].set_xticks(np.arange(0,len(grain_rds),1))
    axes[1,2].set_xticklabels(grain_rds,rotation=30)
    axes[1,2].set_yticks(np.arange(0,len(densities),1))
    axes[1,2].set_yticklabels(densities)
    axes[1,2].set_ylabel('density (kg m-3)')
    axes[1,2].set_xlabel('r_eff (microns)')
    axes[1,2].set_title('M_alg: \n{} ppb'.format(algae[5]))

    fig.tight_layout()

    plt.savefig('/home/joe/Code/Remote_Ice_Surface_Analyser/Manuscript/Figures/BandRatioHeatmaps.jpg',dpi=300)

    # make numpy array of dmensions density x radii
    # iterate through LUT for spectra [dens,radii], calculate band ratio, 
    # and add to correct index in 2D array
    # plot heatmap of 2d array

    return data


def BDA2_of_field_samples():

    """
    2DBA index calculated from field samples after field spectra are averaged over S2 band 4 and 5 wavelengths
    weighted by the sensr spectral response function for each band. The index is then calculated as B5/B4
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

    CIsites = ['21_7_S4', '13_7_SB3', '15_7_S4', '15_7_SB1', '15_7_SB5', '21_7_S2',
            '21_7_SB3', '22_7_S2', '22_7_S4', '23_7_SB1', '23_7_SB2', '23_7_S4',
            '27_7_16_SITE3_WHITE1', '27_7_16_SITE3_WHITE2',
            '27_7_16_SITE2_ICE2', '27_7_16_SITE2_ICE4', '27_7_16_SITE2_ICE6',
            '5_8_16_site2_ice1', '5_8_16_site2_ice2', '5_8_16_site2_ice3',
            '5_8_16_site2_ice4', '5_8_16_site2_ice6', '5_8_16_site2_ice8',
            '5_8_16_site3_ice1', '5_8_16_site3_ice4']

    DF = pd.read_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/RISA_OUT/Spectra_Metadata.csv')

    measured_cells = DF['measured cells/mL']
    modelled_cells = DF['algae_cells/mL']
    BDA2_cells = DF['BDA2_prediction']
    BDA2_cells_ideal = DF['BDA2_ideal_prediction']
    BDA2_cells_S2 = DF['BDA2_S2_prediction']
    BDA2_idx = DF['BDA2_index']
    BDA2_idx_ideal = DF['BDA2_ideal_index']
    BDA2_idx_S2 = DF['BDA2_S2_index']

    BDA2_cells = BDA2_cells[measured_cells>=0]
    BDA2_cells_ideal = BDA2_cells_ideal[measured_cells>=0]
    BDA2_cells_S2 = BDA2_cells_S2[measured_cells>=0]
    BDA2_idx = BDA2_idx[measured_cells>=0]
    BDA2_idx_ideal = BDA2_idx_ideal[measured_cells>=0]
    BDA2_idx_S2 = BDA2_idx_S2[measured_cells>=0]

    modelled_cells = modelled_cells[measured_cells>=0]
    measured_cells = measured_cells[measured_cells>=0]

    ## regression models
    # Ordinary least squares regression
    X = sm.add_constant(modelled_cells)
    model = sm.OLS(measured_cells, modelled_cells).fit()
    summary = model.summary()

    test_x = [0,1000,5000,7500,10000,12500,15000,17500,20000,25000, 30000, 35000, 40000, 50000]
    ypred = model.predict(test_x)    ## imports and data organisation
    import statsmodels.api as sm
    import numpy as np

    CIsites = ['21_7_S4', '13_7_SB3', '15_7_S4', '15_7_SB1', '15_7_SB5', '21_7_S2',
            '21_7_SB3', '22_7_S2', '22_7_S4', '23_7_SB1', '23_7_SB2', '23_7_S4',
            '27_7_16_SITE3_WHITE1', '27_7_16_SITE3_WHITE2',
            '27_7_16_SITE2_ICE2', '27_7_16_SITE2_ICE4', '27_7_16_SITE2_ICE6',
            '5_8_16_site2_ice1', '5_8_16_site2_ice2', '5_8_16_site2_ice3',
            '5_8_16_site2_ice4', '5_8_16_site2_ice6', '5_8_16_site2_ice8',
            '5_8_16_site3_ice1', '5_8_16_site3_ice4']

    DF = pd.read_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/RISA_OUT/Spectra_Metadata.csv')

    measured_cells = DF['measured cells/mL']
    modelled_cells = DF['algae_cells/mL']
    BDA2_cells = DF['BDA2_prediction']
    BDA2_cells_ideal = DF['BDA2_ideal_prediction']
    BDA2_cells_S2 = DF['BDA2_S2_prediction']
    BDA2_idx = DF['BDA2_index']
    BDA2_idx_ideal = DF['BDA2_ideal_index']
    BDA2_idx_S2 = DF['BDA2_S2_index']

    BDA2_cells = BDA2_cells[measured_cells>=0]
    BDA2_cells_ideal = BDA2_cells_ideal[measured_cells>=0]
    BDA2_cells_S2 = BDA2_cells_S2[measured_cells>=0]
    BDA2_idx = BDA2_idx[measured_cells>=0]
    BDA2_idx_ideal = BDA2_idx_ideal[measured_cells>=0]
    BDA2_idx_S2 = BDA2_idx_S2[measured_cells>=0]

    modelled_cells = modelled_cells[measured_cells>=0]
    measured_cells = measured_cells[measured_cells>=0]
    

    ## regression models

    # Ordinary least squares regression
    model = sm.OLS(modelled_cells,measured_cells).fit()
    summary = model.summary()
    test_x = [0,1000,5000,7500,10000,12500,15000,17500,20000,25000, 30000, 35000, 40000, 50000]
    ypred = model.predict(test_x)

    # call function for clean ice comparison
    OutDF = field_vs_SNICAR_NIR()

    # OLS regression for 2BDA predictions
    BDA2_predict = DF['BDA2_prediction']
    BDA2_predict = BDA2_predict[DF['measured cells/mL']>=0]
    BDA2_centre_model = sm.OLS(BDA2_predict,sm.add_constant(measured_cells)).fit()
    BDA2_centre_r2 = np.round(BDA2_centre_model.rsquared,3)

    BDA2_S2_predict = DF['BDA2_S2_prediction']
    BDA2_S2_predict = BDA2_S2_predict[DF['measured cells/mL']>=0]
    BDA2_S2_model = sm.OLS(BDA2_S2_predict,sm.add_constant(measured_cells)).fit()
    BDA2_S2_r2 = np.round(BDA2_S2_model.rsquared,3)

    BDA2_ideal_predict = DF['BDA2_ideal_prediction']
    BDA2_ideal_predict = BDA2_ideal_predict[DF['measured cells/mL']>=0]
    BDA2_ideal_model = sm.OLS(BDA2_ideal_predict,sm.add_constant(measured_cells)).fit()
    BDA2_ideal_r2 = np.round(BDA2_ideal_model.rsquared,3)

    # OLS regression for 2BDA indexes

    BDA2idx_predict = DF['BDA2_index']
    BDA2idx_predict = BDA2idx_predict[DF['measured cells/mL']>=0]
    BDA2idx_centre_model = sm.OLS(BDA2idx_predict,sm.add_constant(measured_cells)).fit()
    BDA2idx_centre_r2 = np.round(BDA2idx_centre_model.rsquared,3)
    BDA2idx_centre_pred = BDA2idx_centre_model.predict(sm.add_constant(test_x))

    BDA2idx_S2_predict = DF['BDA2_S2_index']
    BDA2idx_S2_predict = BDA2idx_S2_predict[DF['measured cells/mL']>=0]
    BDA2idx_S2_model = sm.OLS(BDA2idx_S2_predict,sm.add_constant(measured_cells)).fit()
    BDA2idx_S2_r2 = np.round(BDA2idx_S2_model.rsquared,3)
    BDA2idx_S2_pred = BDA2idx_S2_model.predict(sm.add_constant(test_x))

    BDA2idx_ideal_predict = DF['BDA2_ideal_index']
    BDA2idx_ideal_predict = BDA2idx_ideal_predict[DF['measured cells/mL']>=0]
    BDA2idx_ideal_model = sm.OLS(BDA2idx_ideal_predict,sm.add_constant(measured_cells)).fit()
    BDA2idx_ideal_r2 = np.round(BDA2idx_ideal_model.rsquared,3)
    BDA2idx_ideal_pred = BDA2idx_ideal_model.predict(sm.add_constant(test_x))

    # regress measured cells against band index to give predictive model
       
    test_idx = np.arange(0.99, 1.2, 0.01)
    testModel = sm.OLS(measured_cells,sm.add_constant(BDA2_idx_S2)).fit()
    testModel_r2 = np.round(BDA2idx_S2_model.rsquared,3)
    testModel_pred = testModel.predict(sm.add_constant(BDA2_idx_S2))   

    # call function for clean ice comparison
    OutDF = field_vs_SNICAR_NIR()


    # multipanel figure
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(11,15))

    ax1.plot(measured_cells,color='k',marker = 'o',linestyle = '--',\
        label='field-measured')
    ax1.plot(modelled_cells,color='r',marker = 'x',label='model prediction')
    ax1.set_ylabel('Algal concentration (cells/mL)')    
    ax1.set_xticks(range(len(measured_cells)))
    ax1.set_xticklabels([])
    ax1.legend(loc='best')
    ax1.set_xlabel('Individual samples')
    ax1.set_ylim(0,60000)

    ax2.scatter(measured_cells,modelled_cells, marker='o',color='k')
    ax2.plot(test_x,ypred,linestyle='--')
    ax2.set_ylabel('Algal concentration,\n cells/mL (field)')
    ax2.set_xlabel('Algal concentration,\n clls/mL (predicted by model)')
    ax2.set_xlim(0,50000),ax2.set_ylim(0,60000)
    ax2.text(1000,35000,'r2 = {}\np = {}'.format(np.round(model.rsquared,3),\
        np.round(model.pvalues[0],8)),fontsize=12)

    ax3.plot(OutDF.min_error,linestyle='None', marker='x', color='k')
    ax3.set_ylim(0,0.15)
    ax3.set_xticks(range(len(CIsites)))
    ax3.set_xticklabels([])
    ax3.set_xlabel('Individual Samples')
    ax3.set_ylabel('Absolute error \n(field vs measured NIR spectral albedo)')

    ax4.scatter(measured_cells,BDA2_idx,linestyle='None', marker='x', color='k',\
         label = '2BDA: r2 = {}'.format(np.round(BDA2idx_centre_r2,3)))
    ax4.plot(test_x,BDA2idx_centre_pred,  linestyle='--',color='k')
    ax4.scatter(measured_cells,BDA2_idx_ideal,linestyle='None', marker='o', color='k',\
         facecolor='None', label='2BDA_ideal: r2 = {}'.format(np.round(BDA2idx_ideal_r2,3)))
    ax4.plot(test_x,BDA2idx_ideal_pred, linestyle='-.',color='k',alpha=0.5)
    ax4.scatter(measured_cells,BDA2_idx_S2,linestyle='None', marker='+', color='k',\
         label='2BDA_S2: r2 = {}'.format(np.round(BDA2idx_S2_r2,3)))
    ax4.plot(test_x,BDA2idx_S2_pred, linestyle = 'dotted',color='k')
    ax4.set_ylabel('2BDA band ratio'),ax4.set_xlabel('Measured algal concentration\n (Cells/mL)')
    ax4.set_xlim(0,50000)
    ax4.legend(loc='best')

    ax5.scatter(DF['measured cells/mL'],DF['BDA2_prediction'],marker='o',\
        color='k',facecolor='None',label='2BDA: r2 = {}'.format(BDA2_centre_r2))
    ax5.scatter(DF['measured cells/mL'],DF['BDA2_S2_prediction'],marker='+',\
        color='k', label='2BDA_S2: r2 = {}'.format(BDA2_S2_r2))
    ax5.scatter(DF['measured cells/mL'],DF['BDA2_ideal_prediction'],marker='x',\
        color='k',alpha=1, label='2BDA_ideal: r2 = {}'.format(BDA2_ideal_r2))
    ax5.set_xlabel('Measured algal \nconcentration (cells/mL)')
    ax5.set_ylabel('Algal concentration (cells/mL)\n predicted by 2BDA index')

    ax5.set_ylim(0,50000)
    ax5.legend(loc='best')


    fig.tight_layout()

    savepath = '/home/joe/Code/Remote_Ice_Surface_Analyser/RISA_OUT/Figures_and_Tables'
    fig.savefig(str(savepath+'/measured_modelled_algae.png'),dpi=300)

    return


def field_vs_SNICAR_NIR():
    
    """
    This function takes all clean ice field spectra and compares them to the snicar
    simulated spectra in the NIR wavelengths (0.9 - 1.1 um)

    The out data is stored as a pandas dataframe and saved to csv as 'field_disort_NIR_comparison.csv'

    """

    densities = [350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    grain_rds = [1000, 2000, 3000, 4000, 5000]

    CIsites = ['21_7_S4', '13_7_SB3', '15_7_S4', '15_7_SB1', '15_7_SB5', '21_7_S2',
               '21_7_SB3', '22_7_S2', '22_7_S4', '23_7_SB1', '23_7_SB2', '23_7_S4',
                '27_7_16_SITE3_WHITE1', '27_7_16_SITE3_WHITE2',
               '27_7_16_SITE2_ICE2', '27_7_16_SITE2_ICE4', '27_7_16_SITE2_ICE6',
               '5_8_16_site2_ice1', '5_8_16_site2_ice2', '5_8_16_site2_ice3',
               '5_8_16_site2_ice4', '5_8_16_site2_ice6', '5_8_16_site2_ice8',
               '5_8_16_site3_ice1', '5_8_16_site3_ice4']


    spectra = pd.read_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/Training_Data/HCRF_master_16171819.csv')
    LUT = np.load('/home/joe/Code/BioSNICAR_GO_PY/Spec_LUT_50.npy')
    
    params = []
    # reformat LUT: flatten LUT from 3D to 2D array with one column per combination
    # of RT params, one row per wavelength

    wavelengths = np.arange(0.2,5,0.01)
    
    LUT = LUT[:,:,0,:]
    LUT = LUT.reshape(len(densities)*len(grain_rds),len(wavelengths))

    LUT = LUT[:,70:90] # reduce wavelengths to NIR

    OutDF = pd.DataFrame(columns=['colname','densities','reff','min_error'],index=None)

    densitylist = []
    grainlist = []
    errorlist = []
    collist = []


    for i in np.arange(0,len(spectra.columns),1):

        if (i != 'wavelength') & (spectra.columns[i] in CIsites):

            colname = spectra.columns[i]
            spectrum = np.array(spectra[colname])
            spectrum = spectrum[560:760:10]
            error_array = abs(LUT - spectrum)
            mean_error = np.mean(error_array,axis=1)
            index = np.argmin(mean_error)
            min_error= np.min(mean_error)
            param_idx = np.unravel_index(index,[len(densities),len(grain_rds)])

            densitylist.append(densities[param_idx[0]])
            grainlist.append(grain_rds[param_idx[1]])
            errorlist.append(min_error)
            collist.append(colname)

    OutDF['colname'] = collist
    OutDF['densities'] = densitylist
    OutDF['reff'] = grainlist
    OutDF['min_error'] = errorlist

    #OutDF.to_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/RISA_OUT/Figures_and_Tables/field_disort_NIR_comparison.csv')
    
    fig,ax = plt.subplots()
    ax.plot(OutDF.min_error,linestyle='None',marker='x', color='k'),
    ax.set_ylim(0,0.3)
    ax.set_ylabel('absolute error between modelled \nand measured clean ice NIR albedo)')
    ax.set_xticks(range(len(CIsites))) 
    ax.set_xticklabels([])
    ax.set_xlabel('Samples')
    ax.text(0.2,0.25,'mean absolute error = {}'.format(np.round(OutDF.min_error.mean(),2)))
    ax.text(0.2,0.27,'std of abs error = {}'.format(np.round(OutDF.min_error.std(),2)))

    return OutDF


def WC_experiments():

    """
    *** NOT YET FUNCTIONAL ***

    """

    CIsites = ['21_7_S4', '13_7_SB3', '15_7_S4', '15_7_SB1', '15_7_SB5', '21_7_S2',
               '21_7_SB3', '22_7_S2', '22_7_S4', '23_7_SB1', '23_7_SB2', '23_7_S4']

    BLUEsites = ['WAT_1','WAT_2','WAT_3','WAT_4','WAT_5','WAT_6']

    DISPsites = ['DISP1','DISP2','DISP3','DISP4','DISP5','DISP6','DISP7','DISP8',
    'DISP9','DISP10','DISP11','DISP12','DISP13','DISP14', '27_7_16_DISP1','27_7_16_DISP3']

    spectra = pd.read_csv('/home/joe/Code/Remote_Ice_Surface_Analyser/Training_Data/HCRF_master_16171819.csv')
    LUT = np.load('/home/joe/Code/BioSNICAR_GO_PY/Spec_LUT_50.npy')

    CIspec = spectra[spectra.columns.intersection(CIsites)]
    BLUEspec = spectra[spectra.columns.intersection(BLUEsites)]
    DISPspec = spectra[spectra.columns.intersection(DISPsites)]


    return

savepath = '/home/joe/Code/Remote_Ice_Surface_Analyser/RISA_OUT/Figures_and_Tables/'
compare_predicted_and_measured(savepath)
# clean_ice_field_vs_DISORT_NIR()