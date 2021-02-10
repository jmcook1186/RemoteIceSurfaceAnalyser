"""
Joseph Cook, Aarhus University, Feb 2021

This script contains functions that a) generate 
SNICAR-predicted spectral albedo that approximate
field-measured spectra for a variety of weathering 
crust configurations; b) quantify the albedo change
resulting from a range of WC development scenarios 

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SNICAR_feeder import snicar_feeder
import statsmodels.api as sm  
import collections
import xarray as xr


def find_best_params(field_spectrum, clean=True):

    """
    This function will return the SNICAR parameter set that provides the
    best approximation to a given field spectrum. 

    """

    benchmark = 0.05

    if clean:
        for i in [600, 650, 700, 750, 800, 850, 900, 910]:
            for j in [i-50, i, i+50]:
                for k in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]:
                   
                    params = collections.namedtuple("params","rho_layers, grain_rds, layer_type, dz, mss_cnc_glacier_algae, solzen")
                    params.grain_rds = [i]
                    params.rho_layers = [j]
                    params.layer_type = [1]
                    params.dz = [k]
                    params.mss_cnc_glacier_algae = 0
                    params.solzen = 60

                    albedo, BBA = call_snicar(params)
                    
                    error = abs(albedo[15:230]-field_spectrum)

                    if np.mean(error) < np.mean(benchmark):
                        benchmark = error
                        best_params = (i,j,k)
                        best_albedo = albedo
                
        best_error = benchmark            
    
    else:
        for i in [500, 550, 600, 650, 700, 750, 800, 850, 900, 910]:
            for j in [i-50, i, i+50]:
                for k in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]:
                    for p in [0, 10000, 25000, 50000, 75000, 100000]:#, 125000, 150000, 200000, 250000, 300000]:
                    
                        params = collections.namedtuple("params","rho_layers, grain_rds, layer_type, dz, mss_cnc_glacier_algae, solzen")
                        params.grain_rds = [i,i]
                        params.rho_layers = [j,j]
                        params.layer_type = [1,1]
                        params.dz = [0.001,k]
                        params.mss_cnc_glacier_algae = [p,0]
                        params.solzen = 60

                        albedo, BBA = call_snicar(params)
                        
                        error = abs(albedo[15:230]-field_spectrum)

                        if np.mean(error) < np.mean(benchmark):
                            benchmark = error
                            best_params = (i,j,k,p)
                            best_albedo = albedo
                
        best_error = benchmark    

    return best_params, best_error, best_albedo


def call_snicar(params):

# set dir_base to the location of the BioSNICAR_GO_PY folder

    dir_base = '/home/joe/Code/BioSNICAR_GO_PY/'
    savepath = dir_base # base path for saving figures
       
    TOON = False # toggle Toon et al tridiagonal matrix solver
    ADD_DOUBLE = True # toggle adding-doubling solver
    layer_type = params.layer_type
    DIRECT   = 1        # 1= Direct-beam incident flux, 0= Diffuse incident flux
    APRX_TYP = 1        # 1= Eddington, 2= Quadrature, 3= Hemispheric Mean
    DELTA    = 1        # 1= Apply Delta approximation, 0= No delta
    solzen   = params.solzen      # if DIRECT give solar zenith angle (degrees from 0 = nadir, 90 = horizon)
    rf_ice = 2        # define source of ice refractive index data. 0 = Warren 1984, 1 = Warren 2008, 2 = Picard 2016
    incoming_i = 4
    nbr_lyr = len(params.dz)  # number of snow layers
    R_sfc = 0.1 # reflectance of underlying surface - set across all wavelengths
    rwater = [0]*len(params.dz) # if  using Mie calculations, add radius of optional liquid water coating
    grain_shp =[0]*len(params.dz) # grain shape(He et al. 2016, 2017)
    shp_fctr = [0]*len(params.dz) # shape factor (ratio of aspherical grain radii to that of equal-volume sphere)
    grain_ar = [0]*len(params.dz) # aspect ratio (ratio of width to length)
    side_length = 0
    depth=0
    grain_rds = params.grain_rds
    rho_layers = params.rho_layers
    dz = params.dz
    
    mss_cnc_soot1 = [0]*len(params.dz)    # uncoated black carbon (Bohren and Huffman, 1983)
    mss_cnc_soot2 = [0]*len(params.dz)    # coated black carbon (Bohren and Huffman, 1983)
    mss_cnc_brwnC1 = [0]*len(params.dz)   # uncoated brown carbon (Kirchstetter et al. (2004).)
    mss_cnc_brwnC2 = [0]*len(params.dz)   # sulfate-coated brown carbon (Kirchstetter et al. (2004).)
    mss_cnc_dust1 = [0]*len(params.dz)    # dust size 1 (r=0.05-0.5um) (Balkanski et al 2007)
    mss_cnc_dust2 = [0]*len(params.dz)    # dust size 2 (r=0.5-1.25um) (Balkanski et al 2007)
    mss_cnc_dust3 = [0]*len(params.dz)    # dust size 3 (r=1.25-2.5um) (Balkanski et al 2007)
    mss_cnc_dust4 = [0]*len(params.dz)    # dust size 4 (r=2.5-5.0um)  (Balkanski et al 2007)
    mss_cnc_dust5 = [0]*len(params.dz)    # dust size 5 (r=5.0-50um)  (Balkanski et al 2007)
    mss_cnc_ash1 = [0]*len(params.dz)    # volcanic ash size 1 (r=0.05-0.5um) (Flanner et al 2014)
    mss_cnc_ash2 = [0]*len(params.dz)    # volcanic ash size 2 (r=0.5-1.25um) (Flanner et al 2014)
    mss_cnc_ash3 = [0]*len(params.dz)    # volcanic ash size 3 (r=1.25-2.5um) (Flanner et al 2014)
    mss_cnc_ash4 = [0]*len(params.dz)    # volcanic ash size 4 (r=2.5-5.0um) (Flanner et al 2014)
    mss_cnc_ash5 = [0]*len(params.dz)    # volcanic ash size 5 (r=5.0-50um) (Flanner et al 2014)
    mss_cnc_ash_st_helens = [0]*len(params.dz)   # ash from Mount Saint Helen's
    mss_cnc_Skiles_dust1 = [0]*len(params.dz)    # Colorado dust size 1 (Skiles et al 2017)
    mss_cnc_Skiles_dust2 = [0]*len(params.dz)    # Colorado dust size 2 (Skiles et al 2017)
    mss_cnc_Skiles_dust3 = [0]*len(params.dz)    # Colorado dust size 3 (Skiles et al 2017)
    mss_cnc_Skiles_dust4 = [0]*len(params.dz)  # Colorado dust size 4 (Skiles et al 2017)
    mss_cnc_Skiles_dust5 = [0]*len(params.dz)  # Colorado dust size 5 (Skiles et al 2017)
    mss_cnc_GreenlandCentral1 = [0]*len(params.dz) # Greenland Central dust size 1 (Polashenski et al 2015)
    mss_cnc_GreenlandCentral2 = [0]*len(params.dz) # Greenland Central dust size 2 (Polashenski et al 2015)
    mss_cnc_GreenlandCentral3 = [0]*len(params.dz) # Greenland Central dust size 3 (Polashenski et al 2015)
    mss_cnc_GreenlandCentral4 = [0]*len(params.dz) # Greenland Central dust size 4 (Polashenski et al 2015)
    mss_cnc_GreenlandCentral5 = [0]*len(params.dz) # Greenland Central dust size 5 (Polashenski et al 2015)
    mss_cnc_Cook_Greenland_dust_L = [0]*len(params.dz)
    mss_cnc_Cook_Greenland_dust_C = [0]*len(params.dz)
    mss_cnc_Cook_Greenland_dust_H = [0]*len(params.dz)
    mss_cnc_snw_alg = [0]*len(params.dz)    # Snow Algae (spherical, C nivalis) (Cook et al. 2017)
    mss_cnc_glacier_algae = params.mss_cnc_glacier_algae   # glacier algae type1 (Cook et al. 2020)

    nbr_aer = 30

    # Set names of files containing the optical properties of these LAPs:
    FILE_soot1  = 'mie_sot_ChC90_dns_1317.nc'
    FILE_soot2  = 'miecot_slfsot_ChC90_dns_1317.nc'
    FILE_brwnC1 = 'brC_Kirch_BCsd.nc'
    FILE_brwnC2 = 'brC_Kirch_BCsd_slfcot.nc'
    FILE_dust1  = 'dust_balkanski_central_size1.nc'
    FILE_dust2  = 'dust_balkanski_central_size2.nc'
    FILE_dust3  = 'dust_balkanski_central_size3.nc'
    FILE_dust4  = 'dust_balkanski_central_size4.nc'
    FILE_dust5 = 'dust_balkanski_central_size5.nc'
    FILE_ash1  = 'volc_ash_eyja_central_size1.nc'
    FILE_ash2 = 'volc_ash_eyja_central_size2.nc'
    FILE_ash3 = 'volc_ash_eyja_central_size3.nc'
    FILE_ash4 = 'volc_ash_eyja_central_size4.nc'
    FILE_ash5 = 'volc_ash_eyja_central_size5.nc'
    FILE_ash_st_helens = 'volc_ash_mtsthelens_20081011.nc'
    FILE_Skiles_dust1 = 'dust_skiles_size1.nc'
    FILE_Skiles_dust2 = 'dust_skiles_size2.nc'
    FILE_Skiles_dust3 = 'dust_skiles_size3.nc'
    FILE_Skiles_dust4 = 'dust_skiles_size4.nc'
    FILE_Skiles_dust5 = 'dust_skiles_size5.nc'
    FILE_GreenlandCentral1 = 'dust_greenland_central_size1.nc'
    FILE_GreenlandCentral2 = 'dust_greenland_central_size2.nc'
    FILE_GreenlandCentral3 = 'dust_greenland_central_size3.nc'
    FILE_GreenlandCentral4 = 'dust_greenland_central_size4.nc'
    FILE_GreenlandCentral5  = 'dust_greenland_central_size5.nc'
    FILE_Cook_Greenland_dust_L = 'dust_greenland_Cook_LOW_20190911.nc'
    FILE_Cook_Greenland_dust_C = 'dust_greenland_Cook_CENTRAL_20190911.nc'
    FILE_Cook_Greenland_dust_H = 'dust_greenland_Cook_HIGH_20190911.nc'
    FILE_snw_alg  = 'snw_alg_r025um_chla020_chlb025_cara150_carb140.nc'
    FILE_glacier_algae = 'Glacier_Algae_480.nc'


    #######################################
    # IF NO INPUT ERRORS --> FUNCTION CALLS
    #######################################

        
    [wvl, albedo, BBA, BBAVIS, BBANIR, abs_slr, heat_rt] =\
    snicar_feeder(dir_base,\
    rf_ice, incoming_i, DIRECT, layer_type,\
    APRX_TYP, DELTA, solzen, TOON, ADD_DOUBLE, R_sfc, dz, rho_layers, grain_rds,\
    side_length, depth, rwater, nbr_lyr, nbr_aer, grain_shp, shp_fctr, grain_ar,\
    mss_cnc_soot1, mss_cnc_soot2, mss_cnc_brwnC1, mss_cnc_brwnC2, mss_cnc_dust1,\
    mss_cnc_dust2, mss_cnc_dust3, mss_cnc_dust4, mss_cnc_dust5, mss_cnc_ash1, mss_cnc_ash2,\
    mss_cnc_ash3, mss_cnc_ash4, mss_cnc_ash5, mss_cnc_ash_st_helens, mss_cnc_Skiles_dust1, mss_cnc_Skiles_dust2,\
    mss_cnc_Skiles_dust3, mss_cnc_Skiles_dust4, mss_cnc_Skiles_dust5, mss_cnc_GreenlandCentral1,\
    mss_cnc_GreenlandCentral2, mss_cnc_GreenlandCentral3, mss_cnc_GreenlandCentral4,\
    mss_cnc_GreenlandCentral5, mss_cnc_Cook_Greenland_dust_L, mss_cnc_Cook_Greenland_dust_C,\
    mss_cnc_Cook_Greenland_dust_H, mss_cnc_snw_alg, mss_cnc_glacier_algae, FILE_soot1,\
    FILE_soot2, FILE_brwnC1, FILE_brwnC2, FILE_dust1, FILE_dust2, FILE_dust3, FILE_dust4, FILE_dust5,\
    FILE_ash1, FILE_ash2, FILE_ash3, FILE_ash4, FILE_ash5, FILE_ash_st_helens, FILE_Skiles_dust1, FILE_Skiles_dust2,\
    FILE_Skiles_dust3, FILE_Skiles_dust4, FILE_Skiles_dust5, FILE_GreenlandCentral1,\
    FILE_GreenlandCentral2, FILE_GreenlandCentral3, FILE_GreenlandCentral4, FILE_GreenlandCentral5,\
    FILE_Cook_Greenland_dust_L, FILE_Cook_Greenland_dust_C, FILE_Cook_Greenland_dust_H, FILE_snw_alg, FILE_glacier_algae)

    return albedo, BBA


def match_field_spectra(field_data_fname, fnames, rho, rds, dz, alg, measured_cells,\
    CIsites, LAsites, HAsites, savepath):
    
    """
    plot field against SNICAR spectra
    requires parameters to be known in advance and hard coded inside this function
    the relevant params can be generated using the find_best_params() func

    params:
    field_data_fname = filename for spectral database
    
    The following params are parallel arrays - the order matters and 
    must match the filenames! Pairs of values represent values for 
    upper and lower layers in model. These are the values used to
    generate snicar spectrum to match field spectrum with name = fname[i]
    
    fnames = sample IDs for spectra to match model runs
    rho = pairs of density values [upper, lower]
    rds = pairs of r_eff values [upper, lower]
    dz = layer thickness [upper, lower] NB. upper is always 0.001
    alg = mass concentration of algae [upper, lower] NB. lower is always 0
    measured_cells = array of the actual measured cell concentration for each spectrum

    e.g.
    fnames= ['2016_WI_8','14_7_SB6','14_7_SB9','14_7_SB1','21_7_SB2','14_7_SB2', '22_7_SB3', 'RAIN']
    rho = [[550,550],[650,650],[800,800],[850,850],[750,750],[800,800],[800,800],[900,900]]
    rds = [[550,550],[650,650],[850,850],[850,850],[800,800],[800,800],[750,750],[900,900]]
    dz = [[0.001,0.3],[0.001,0.09],[0.001,0.03],[0.001,0.03],[0.001,0.02],[0.001,0.06],[0.001,0.05],[0.001,0.03]]
    alg = [[0,0],[0,0],[20000,0],[30000,0],[45000,0],[3000,0],[8000,0],[0,0]]  

    returns:
    None, but saves figure to savepath

    """

    spectra = pd.read_csv(field_data_fname)

    # reformat feld spectra to match snicar resolution
    spectra = spectra[::10]

    # gather spectra for each surface type
    CIspec = spectra[spectra.columns.intersection(CIsites)]
    HAspec = spectra[spectra.columns.intersection(HAsites)]
    LAspec = spectra[spectra.columns.intersection(LAsites)]
    RAINspec = spectra['RAIN2']

    # define local function for calling snicar
    def simulate_albedo(rds, rho, dz, alg):
  
        params = collections.namedtuple("params",\
            "rho_layers, grain_rds, layer_type, dz,\
                 mss_cnc_glacier_algae, solzen")
        params.grain_rds = rds
        params.rho_layers = rho
        params.layer_type = [1,1]
        params.dz = dz
        params.mss_cnc_glacier_algae = alg
        params.solzen = 53
        albedo, BBA = call_snicar(params)
  
        return albedo, BBA
  
    # calculate Malg in cells/mL from pbb for figure labels
    # assumes cells have radius 4 um, lentgh 40 microns
    alg_conc = []
    for i in np.arange(0,len(alg),1):
        alg_conc.append(int(alg[i][0] / ((((((np.pi*4**2)*40)*0.0014)*0.3)/0.917))))
    
    # set up output array
    # and call snicar with each set of params
    OutArray = np.zeros(shape=(len(fnames),480))

    for i in np.arange(0,len(fnames),1):
        albedo,BBA=simulate_albedo(rds[i],rho[i],dz[i],alg[i])
        OutArray[i,:] = albedo
        print(BBA)


    # calculate mean absolute error for model vs measured spectrum
    error = []
    for i in np.arange(0,len(fnames),1):
        error.append(abs(np.mean(spectra[fnames[i]] - OutArray[i,15:230])))


    # plot figure
    fig,axes = plt.subplots(4,2,figsize=(10,10))

    axes[0,0].plot(spectra.Wavelength[5:],CIspec['WI_8'][5:],label='field')
    axes[0,0].plot(spectra.Wavelength,OutArray[0,15:230],label='model',linestyle='--')
    axes[0,0].set_ylim(0,1), axes[0,0].set_xlim(350,1800)
    axes[0,0].text(1400,0.2,"r_eff: {}\nrho: {}\nM_alg: {}\ndz: {}\nerror: {:.3f}".format(\
        rds[0][0],rho[0][0],alg_conc[0],dz[0][1],error[0]))
    axes[0,0].text(400,0.1,'{} \n{} cells/mL '.format(fnames[0],measured_cells[0]))
    axes[0,0].set_ylabel('Albedo'), axes[0,0].set_xlabel('Wavelength (nm)')
    axes[0,0].legend(loc='best')

    axes[0,1].plot(spectra.Wavelength,CIspec['14_7_SB6'],label='field')
    axes[0,1].plot(spectra.Wavelength,OutArray[1,15:230],label='model',linestyle='--')
    axes[0,1].set_ylim(0,1), axes[0,1].set_xlim(350,1800)
    axes[0,1].text(1450,0.3,"r_eff: {}\nrho: {}\nM_alg: {}\ndz: {}\nerror: {:.3f}".format(\
        rds[1][0],rho[1][0],alg_conc[1],dz[1][1],error[1]))
    axes[0,1].text(400,0.8,'{}: \n{} cells/mL'.format(fnames[1],measured_cells[1]))
    axes[0,1].set_ylabel('Albedo'), axes[0,1].set_xlabel('Wavelength (nm)')

    axes[1,0].plot(spectra.Wavelength,HAspec['14_7_SB9'])
    axes[1,0].plot(spectra.Wavelength,OutArray[2,15:230],linestyle='--')
    axes[1,0].set_ylim(0,1), axes[1,0].set_xlim(350,1800)
    axes[1,0].text(1400,0.3,"r_eff: {}\nrho: {}\nM_alg: {}\ndz: {}\nerror: {:.3f}".format(\
        rds[2][0],rho[2][0],alg_conc[2],dz[2][1],error[2]))
    axes[1,0].text(400,0.8,'{}: \n{} cells/mL'.format(fnames[2],measured_cells[2]))
    axes[1,0].set_ylabel('Albedo'), axes[1,0].set_xlabel('Wavelength (nm)')

    axes[1,1].plot(spectra.Wavelength,HAspec['14_7_SB1'])
    axes[1,1].plot(spectra.Wavelength,OutArray[3,15:230],linestyle='--')
    axes[1,1].set_ylim(0,1), axes[1,1].set_xlim(350,1800)
    axes[1,1].text(1400,0.3,"r_eff: {}\nrho: {}\nM_alg: {}\ndz: {}\nerror: {:.3f}".format(\
        rds[3][0],rho[3][0],alg_conc[3],dz[3][1],error[3]))
    axes[1,1].text(400,0.8,'{}: \n{} cells/mL'.format(fnames[3],measured_cells[3]))
    axes[1,1].set_ylabel('Albedo'), axes[1,1].set_xlabel('Wavelength (nm)')

    axes[2,0].plot(spectra.Wavelength,HAspec['21_7_SB2'])
    axes[2,0].plot(spectra.Wavelength,OutArray[4,15:230],linestyle='--')
    axes[2,0].set_ylim(0,1), axes[2,0].set_xlim(350,1800)
    axes[2,0].text(1400,0.3,"r_eff: {}\nrho: {}\nM_alg: {}\ndz: {}\nerror: {:.3f}".format(\
        rds[4][0],rho[4][0],alg_conc[4],dz[4][1], error[4]))
    axes[2,0].text(400,0.8,'{}: \n{} cells/mL'.format(fnames[4],measured_cells[4]))
    axes[2,0].set_ylabel('Albedo'), axes[2,0].set_xlabel('Wavelength (nm)')

    axes[2,1].plot(spectra.Wavelength,LAspec['14_7_SB2'])
    axes[2,1].plot(spectra.Wavelength,OutArray[5,15:230],linestyle='--')
    axes[2,1].set_ylim(0,1), axes[2,1].set_xlim(350,1800)
    axes[2,1].text(1400,0.3,"r_eff: {}\nrho: {}\nM_alg: {}\ndz: {}\nerror: {:.3f}".format(\
        rds[5][0],rho[5][0],alg_conc[5],dz[5][1],error[5]))
    axes[2,1].text(400,0.8,'{}: \n{} cells/mL'.format(fnames[5],measured_cells[5]))
    axes[2,1].set_ylabel('Albedo'), axes[2,1].set_xlabel('Wavelength (nm)')

    axes[3,0].plot(spectra.Wavelength,LAspec['22_7_SB3'])
    axes[3,0].plot(spectra.Wavelength,OutArray[6,15:230],linestyle='--')
    axes[3,0].set_ylim(0,1), axes[3,0].set_xlim(350,1800)
    axes[3,0].text(1400,0.3,"r_eff: {}\nrho: {}\nM_alg: {}\ndz: {}\nerror: {:.3f}".format(\
        rds[6][0],rho[6][0],alg_conc[6],dz[6][1],error[6]))
    axes[3,0].text(400,0.8,'{}: \n{} cells/mL'.format(fnames[6],measured_cells[6]))
    axes[3,0].set_ylabel('Albedo'), axes[3,0].set_xlabel('Wavelength (nm)')

    axes[3,1].plot(spectra.Wavelength,RAINspec)
    axes[3,1].plot(spectra.Wavelength,OutArray[7,15:230],linestyle='--')
    axes[3,1].set_ylim(0,1), axes[3,1].set_xlim(350,1800)
    axes[3,1].text(1400,0.3,"r_eff: {}\nrho: {}\nM_alg: {}\ndz: {}\nerror: {:.3f}".format(\
        rds[7][0],rho[7][0],alg_conc[7],dz[7][1],error[7]))
    axes[3,1].text(400,0.8,'{}: \n{} cells/mL'.format(fnames[1],measured_cells[1]))
    axes[3,1].set_ylabel('Albedo'), axes[3,1].set_xlabel('Wavelength (nm)')

    fig.tight_layout()
    plt.savefig(str(savepath+'RTMvsField.jpg'),dpi=300)

    return



def isolate_biological_effect(field_data_fname, CIsites, LAsites, HAsites, savepath):

    """
    This function estimates the albedo reduction resulting from the ice physical changes
    versus the biological growth. 

    Some nuance to the interpretation because the ce surface likely would not
    degrade to the same extent without the algal bloom.

    """

    #read in spectral database
    spectra = pd.read_csv(field_data_fname)

    # reformat feld spectra to match snicar resolution
    spectra = spectra[::10]
    CIspec = spectra[spectra.columns.intersection(CIsites)]
    HAspec = spectra[spectra.columns.intersection(HAsites)]
    LAspec = spectra[spectra.columns.intersection(LAsites)] 
    meanCI = CIspec.mean(axis=1)
    meanHA = HAspec.mean(axis=1)
    meanLA = LAspec.mean(axis=1)

    # define local function for calling snicar
    def simulate_albedo(rds, rho, dz, alg):
        params = collections.namedtuple("params","rho_layers, grain_rds, layer_type, dz, mss_cnc_glacier_algae, solzen")
        params.grain_rds = rds
        params.rho_layers = rho
        params.layer_type = [1,1]
        params.dz = dz
        params.mss_cnc_glacier_algae = alg
        params.solzen = 53
        albedo, BBA = call_snicar(params)
        return albedo, BBA
    
    # call snicar to generat esimulated spectrum
    SNICARalbedo, BBA = simulate_albedo([850,850],[800,800],[0.001,0.03],[0,0])
    SNICARalbedo = SNICARalbedo[15:230]

    # plot figure
    x = np.arange(350,2500,10)
    fig, (ax1) = plt.subplots(1,1,figsize=(5,5))
    ax1.plot(x,meanCI,linestyle='--',alpha = 0.4,label='Clean ice (mean)')
    ax1.plot(x,meanHA,linestyle='-.',alpha = 0.4,label='Algal ice (mean)')
    ax1.plot(x,SNICARalbedo,linestyle='dotted',alpha = 0.4,label='Clean ice (model)')
    ax1.fill_between(x,meanCI,SNICARalbedo,alpha=0.2)
    ax1.fill_between(x,SNICARalbedo,meanHA,color='k',alpha=0.2)
    ax1.set_xlim(350,1500), ax1.legend(loc='best')
    ax1.set_ylabel('Albedo'), ax1.set_xlabel('Wavelength (nm)')

    plt.savefig(str(savepath+'/BiovsPhysEffect.jpg'),dpi=300)
        
    # define incoming to calculate broadband albedo
    incoming = xr.open_dataset('/home/joe/Code/BioSNICAR_GO_PY/Data/Mie_files/480band/fsds/swnb_480bnd_sas_clr_SZA60.nc')
    incoming = incoming['flx_frc_sfc'].values
    incoming = incoming[15:230]

    # calculate broadband albedo of each case
    HA_BBA = np.sum(meanHA*incoming)/np.sum(incoming)
    CI_BBA = np.sum(meanCI*incoming)/np.sum(incoming)
    CI2_BBA = np.sum(SNICARalbedo*incoming)/np.sum(incoming)

    # calculate change due to bio/phys as BBA difference
    delAbio = CI2_BBA-HA_BBA
    delAphys = CI_BBA-CI2_BBA

    return delAbio,delAphys

