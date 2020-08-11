"""
This script compares takes field-measured spectra where metadata is available and subtracts
the reflectance at 9 key wavelengths to the same wavelengths in a LUT of DISORT 
simulated spectra. The combination with the lowest absolute error is selected and the 
parameters used in DISORT to generate the best-matching spectra are added to a list.

This enables manual validation of the LUT-comparison method.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def grab_params():

    """
    This function takes all field spectra and compares them to the DISORT simulated spectra
    LUT used in the Big Ice Surf Classifier. 

    The resulting predicted params have been collated with the site metadata to be loaded 
    in future instead of generating the data again using this function.

    Results available in: /home/joe/Code/BigIceSurfClassifier/Spectra_Metadata.csv


    NB to obtain LUT param predictions for specific field spectra:

    spectra.columns.get_loc('21_7_SB3')
    
    >> 46

    params[46]

    >>
    ([700, 700, 700, 700, 700],
    [8000, 8000, 8000, 8000, 8000],
    [20000, 0, 0, 0, 0])

    """

    densities = [[400,400,400,400,400],[450,450,450,450,450],[500,500,500,500,500],\
        [550,550,550,550,550],[600,600,600,600,600],[650,650,650,650,650],\
            [700,700,700,700,700],[750,750,750,750,750],[800,800,800,800,800],\
                [850,850,850,850,850],[900,900,900,900,900]]

    side_lengths = [[500,500,500,500,500],[700,700,700,700,700],[900,900,900,900,900],[1100,1100,1100,1100,1100],
    [1300,1300,1300,1300,1300],[1500,1500,1500,1500,1500],[2000,2000,2000,2000,2000],[3000,3000,3000,3000,3000],
    [5000,5000,5000,5000,5000],[8000,8000,8000,8000,8000],[10000,10000,10000,10000,10000],
    [15000,15000,15000,15000,15000]]

    algae = [[0,0,0,0,0], [1000,0,0,0,0], [5000,0,0,0,0], [10000,0,0,0,0], [50000,0,0,0,0], [10000,0,0,0,0],\
        [15000,0,0,0,0], [20000,0,0,0,0], [250000,0,0,0,0], [50000,0,0,0,0], [75000,0,0,0,0], [100000,0,0,0,0],\
            [125000,0,0,0,0], [150000,0,0,0,0], [1750000,0,0,0,0], [200000,0,0,0,0], [250000,0,0,0,0]]



    spectra = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/Training_Data/HCRF_master_16171819.csv')
    LUT = np.load('/home/joe/Code/BigIceSurfClassifier/Process_Dir/LUT_cz05.npy')
    params = []
    # reformat LUT: flatten LUT from 3D to 2D array with one column per combination
    # of RT params, one row per wavelength

    wavelengths = np.arange(0.3,5,0.01)
    LUT = LUT.reshape(2244,len(wavelengths))
    idx = [19, 26, 36, 40, 44, 48, 56, 131, 190]
    LUT = LUT[:,idx] # reduce wavelengths to only the 9 that match the S2 image

    for i in np.arange(0,len(spectra.columns),1):
        
        if i != 'wavelength':
            colname = spectra.columns[i]
            spectrum = np.array(spectra[colname][idx])
            error_array = abs(LUT - spectrum)
            mean_error = np.mean(error_array,axis=1)
            index = np.argmin(mean_error)
            param_idx = np.unravel_index(index,[len(densities),len(side_lengths),len(algae)])

            params.append((densities[param_idx[0]],side_lengths[param_idx[1]],algae[param_idx[2]]))
    

    return


def compare_predicted_and_measured(savepath):

    # load metadata with predicted params
    DF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Spectra_Metadata.csv')

    # divide by surface class and select algal concentration column
    HA_alg = DF['Algae'][DF['Surf Type']=='HA']
    LA_alg = DF['Algae'][DF['Surf Type']=='LA']
    CI_alg = DF['Algae'][DF['Surf Type']=='CI']
    SN_alg = DF['Algae'][DF['Surf Type']=='SNOW']

    HA_dns = DF['Density'][DF['Surf Type']=='HA']
    LA_dns = DF['Density'][DF['Surf Type']=='LA']
    CI_dns = DF['Density'][DF['Surf Type']=='CI']
    SN_dns = DF['Density'][DF['Surf Type']=='SNOW']

    HA_grn = DF['Grain Size'][DF['Surf Type']=='HA']
    LA_grn = DF['Grain Size'][DF['Surf Type']=='LA']
    CI_grn = DF['Grain Size'][DF['Surf Type']=='CI']
    SN_grn = DF['Grain Size'][DF['Surf Type']=='SNOW']

    data_alg = [HA_alg,LA_alg,CI_alg,SN_alg]
    data_dns = [HA_dns, LA_dns, CI_dns, SN_dns]
    data_grn = [HA_grn, LA_grn, CI_grn, SN_grn]

    
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,15))
    
    ax1.boxplot(data_alg,whis='range')
    ax1.set_xticklabels(['HA','LA','CI','SN'])
    ax1.set_ylabel('Predicted algal Concentration (ppb)')
    ax1.set_xlabel('Surface class from field notes')
    
    ax2.boxplot(data_dns,whis='range')
    ax2.set_xticklabels(['HA','LA','CI','SN'])
    ax2.set_ylabel('Predicted ice density (kg m-3)')
    ax2.set_xlabel('Surface class from field notes')

    ax3.boxplot(data_grn,whis='range')
    ax3.set_xticklabels(['HA','LA','CI','SN'])
    ax3.set_ylabel('Predicted grain size (microns)')
    ax3.set_xlabel('Surface class from field notes')

    fig.tight_layout()
    plt.savefig(str(savepath+'Predicted_algae_by_class.png'),dpi=300)

    return


def compare_measured_concn_to_predicted(savepath):

    DF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Spectra_Metadata.csv')

    measured_ppb = DF['cells/mL_calcd']
    modelled_ppb = DF['Algae']

    # grab measured cells/mL converted to ppb (predicted)
    # and 
    modelled_ppb = modelled_ppb[measured_ppb>0]
    measured_ppb = measured_ppb[measured_ppb>0]
    

    import statsmodels.api as sm
    import numpy as np

    # Ordinary least squares regression
    model = sm.OLS(measured_ppb, modelled_ppb).fit()
    summary = model.summary()

    test_x = [0,10000,50000,75000,100000,125000,150000,175000,200000,250000]
    ypred = model.predict(test_x)

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))

    ax1.plot(measured_ppb,color='k',marker = 'o',linestyle = '--',label='measured')
    ax1.plot(modelled_ppb,color='r',marker = 'x',label='modelled')
    ax1.set_ylabel('Algal concentration (ppb)')
    ax1.legend(loc='best')

    ax2.scatter(modelled_ppb,measured_ppb, marker='o',color='k')
    ax2.plot(test_x,ypred,linestyle='--')
    ax2.set_ylabel('Algal concentration, ppb (field)')
    ax2.set_xlabel('Algal concentration, ppb (predicted by model)')
    ax2.text(0,350000,'r2 = {}\np = {}'.format(np.round(model.rsquared,3),np.round(model.pvalues[0],8)),fontsize=12)

    fig.tight_layout()

    savepath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Figures_and_Tables'
    fig.savefig(str(savepath+'/measured_modelled_algae.png'),dpi=300)

    return


def combined_figure(savepath):

    import statsmodels.api as sm
    import numpy as np

    # load metadata with predicted params
    DF = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Spectra_Metadata.csv')

    # divide by surface class and select algal concentration column
    HA_alg = DF['Algae'][DF['Surf Type']=='HA']
    LA_alg = DF['Algae'][DF['Surf Type']=='LA']
    CI_alg = DF['Algae'][DF['Surf Type']=='CI']
    SN_alg = DF['Algae'][DF['Surf Type']=='SNOW']

    HA_dns = DF['Density'][DF['Surf Type']=='HA']
    LA_dns = DF['Density'][DF['Surf Type']=='LA']
    CI_dns = DF['Density'][DF['Surf Type']=='CI']
    SN_dns = DF['Density'][DF['Surf Type']=='SNOW']

    HA_grn = DF['Grain Size'][DF['Surf Type']=='HA']
    LA_grn = DF['Grain Size'][DF['Surf Type']=='LA']
    CI_grn = DF['Grain Size'][DF['Surf Type']=='CI']
    SN_grn = DF['Grain Size'][DF['Surf Type']=='SNOW']

    data_alg = [HA_alg,LA_alg,CI_alg,SN_alg]
    data_dns = [HA_dns, LA_dns, CI_dns, SN_dns]
    data_grn = [HA_grn, LA_grn, CI_grn, SN_grn]


    DF2 = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Spectra_Metadata.csv')

    measured_ppb = DF2['ppb_calc_frm_cllspermL']
    modelled_ppb = DF2['Algae']

    # grab measured cells/mL converted to ppb (predicted)
    # and 
    modelled_ppb = modelled_ppb[measured_ppb>0]
    measured_ppb = measured_ppb[measured_ppb>0]

    # Ordinary least squares regression
    model = sm.OLS(measured_ppb, modelled_ppb).fit()
    summary = model.summary()

    test_x = [0,10000,50000,75000,100000,125000,150000,175000,200000,250000]
    ypred = model.predict(test_x)


    fig, (ax1,ax2, ax3) = plt.subplots(3,1,figsize=(10,10))

    ax1.plot(measured_ppb,color='k',marker = 'o',linestyle = '--',label='measured')
    ax1.plot(modelled_ppb,color='r',marker = 'x',label='modelled')
    ax1.set_ylabel('Algal concentration (ppb)')
    ax1.legend(loc='best')

    ax2.scatter(modelled_ppb,measured_ppb, marker='o',color='k')
    ax2.plot(test_x,ypred,linestyle='--')
    ax2.set_ylabel('Algal concentration, ppb (field)')
    ax2.set_xlabel('Algal concentration, ppb (predicted by model)')
    ax2.text(0,350000,'r2 = {}\np = {}'.format(np.round(model.rsquared,3),np.round(model.pvalues[0],8)),fontsize=12)

    ax3.boxplot(data_alg,whis='range')
    ax3.set_xticklabels(['HA','LA','CI','SN'])
    ax3.set_ylabel('Predicted algal Concentration (ppb)')
    ax3.set_xlabel('Surface class from field notes')

    plt.savefig(str(savepath+'FieldValidationFig.png'),dpi=300)

    return

savepath = '/home/joe/Code/BigIceSurfClassifier/BISC_OUT/Figures_and_Tables/'
compare_predicted_and_measured(savepath)
compare_measured_concn_to_predicted(savepath)
combined_figure(savepath) 

# this function just takes elements from the other two functions
# to create a 3 panel figure with 1 plot from func 1 and 2 plots from func 2.




def clean_ice_field_vs_DISORT():
    
    """
    This function takes all clean ice field spectra and compares them to the DISORT
    simulated spectra. 

    Results available in: /home/joe/Code/BigIceSurfClassifier/Spectra_Metadata.csv

    NB to obtain LUT param predictions for specific field spectra:

    spectra.columns.get_loc('21_7_SB3')
    
    >> 46

    params[46]

    >>
    ([700, 700, 700, 700, 700],
    [8000, 8000, 8000, 8000, 8000],
    [20000, 0, 0, 0, 0])

    """

    densities = [400,450,500,550,600,650,700,750,800,850,900]

    grainsize = [500,700,900,1100,1300,1500,2000,3000,5000,8000,10000,15000,20000,25000,30000]


    CIsites = ['21_7_S4', '13_7_SB3', '15_7_S4', '15_7_SB1', '15_7_SB5', '21_7_S2',
               '21_7_SB3', '22_7_S2', '22_7_S4', '23_7_SB1', '23_7_SB2', '23_7_S4',
               'WI_1', 'WI_2', 'WI_4', 'WI_5', 'WI_6', 'WI_7', 'WI_9', 'WI_10', 'WI_11',
               'WI_12', 'WI_13', '27_7_16_SITE3_WHITE1', '27_7_16_SITE3_WHITE2',
               '27_7_16_SITE2_ICE2', '27_7_16_SITE2_ICE4', '27_7_16_SITE2_ICE6',
               '5_8_16_site2_ice1', '5_8_16_site2_ice2', '5_8_16_site2_ice3',
               '5_8_16_site2_ice4', '5_8_16_site2_ice6', '5_8_16_site2_ice8',
               '5_8_16_site3_ice1', '5_8_16_site3_ice4', 
                'fox11_25_',	'fox11_2_', 'fox11_7_', 'fox11_8_', 'fox13_1b_', 'fox13_2_',
                'fox13_2a_', 'fox13_2b_', 'fox13_3_', 'fox13_3a_',
               'fox13_3b_', 'fox13_6a_', 'fox13_7_', 'fox13_7a_', 'fox13_7b_', 'fox13_8_',	
               'fox13_8a_', 'fox13_8b_', 'fox14_2b_', 'fox14_3_', 'fox14_3a_', 'fox17_8_',
               'fox17_8a_', 'fox17_8b_', 'fox17_9b_', 'fox24_17_']


    spectra = pd.read_csv('/home/joe/Code/BigIceSurfClassifier/Training_Data/HCRF_master_16171819.csv')
    LUT = np.load('/home/joe/Code/BigIceSurfClassifier/Process_Dir/LUT_clean.npy')
    params = []
    # reformat LUT: flatten LUT from 3D to 2D array with one column per combination
    # of RT params, one row per wavelength

    wavelengths = np.arange(0.3,5,0.01)
    
    LUT = LUT.reshape(len(densities)*len(grainsize),len(wavelengths))

    LUT = LUT[:,60:80] # reduce wavelengths to NIR

    OutDF = pd.DataFrame(columns=['colname','densities','grainsize','min_error'],index=None)

    densitylist = []
    grainlist = []
    errorlist = []
    collist = []


    for i in np.arange(0,len(spectra.columns),1):

        if (i != 'wavelength') & (spectra.columns[i] in CIsites):

            colname = spectra.columns[i]
            spectrum = np.array(spectra[colname])
            spectrum = spectrum[550:750:10]
            error_array = abs(LUT - spectrum)
            mean_error = np.mean(error_array,axis=1)
            index = np.argmin(mean_error)
            min_error= np.min(mean_error)
            param_idx = np.unravel_index(index,[len(densities),len(grainsize)])

            densitylist.append(densities[param_idx[0]])
            grainlist.append(grainsize[param_idx[1]])
            errorlist.append(min_error)
            collist.append(colname)

    OutDF['colname'] = collist
    OutDF['densities'] = densitylist
    OutDF['grainsize'] = grainlist
    OutDF['min_error'] = errorlist

    return