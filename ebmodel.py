#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:41:08 2018

@author: joe
"""

import numpy as np
import math


def calc_dayang(day):
    return (day / 365.25) * 360



def calc_eqtim(dayang):
    return (-0.128 * np.sin((dayang - 2.8) * (math.pi / 180))) -    (0.165 * np.sin(((2 * dayang) + 19.7) * (math.pi / 180)))



def calc_solhour(time, lon, lon_ref, eqtim, summertime):
    return (time / 100) + ((lon - lon_ref) /15) + eqtim - summertime



def calc_soldec(day):
    i1 = -23.2559 * np.cos(((2 * math.pi * day) / 365) + 0.1582)
    i2 = -0.3915 * np.cos(((4 * math.pi * day)/ 365) + 0.0934)
    i3 = -0.1764 * np.cos(((6 * math.pi * day) / 365) + 0.4539)
    return 0.3948 + (i1 + i2 + i3)



def calc_solhran(solhour):
    return 15 * (solhour - 12)



def calc_solaltr(lat, soldec, solhran):
    return np.arcsin(( 
        ((np.sin(lat*(math.pi/180))) * (np.sin(soldec*(math.pi/180)))) +
        ((np.cos(lat*(math.pi/180))) * (np.cos(soldec*(math.pi/180))) * (np.cos(solhran*(math.pi/180)))) 
        ))



def calc_solaltd(solaltr):
    return solaltr * 180/ math.pi



def calc_cossolaz(lat, solaltr, soldec):
    return ( ((np.sin(lat*(math.pi/180)))*(np.sin(solaltr))) - ((np.sin(soldec*(math.pi/180)))) ) / ((np.cos((lat)*(math.pi/180)))*((np.cos(solaltr))))



def calc_sinolaz(soldec, solhran, solaltr):
    return (np.cos(soldec*(math.pi / 180)))*(np.sin(solhran*(math.pi/180)))/(np.cos(solaltr))



def calc_solaz(sinsolaz, cossolaz):
    if sinsolaz < 0:
        return -np.arccos(cossolaz)
    else:
        return np.arccos(cossolaz)



def calc_solaz360(solaz):
    return (solaz*180/math.pi)+180



def calc_cloudn(inswrad, elevation, solaltr):
    """ Calculate cloud cover
    B+A eqn. 3
    """
    return np.min([1-(inswrad/((1368*(0.75**(100000/(100000*(1-(elevation*0.0001)))))*np.cos(1.57-solaltr)))), 1])



def calc_diffuser(cloudn):
    """ Calculate diffuse fraction of total incoming shortwave radiation 
    B+A eqn. 2
    """
    if cloudn > 0.8:
        return 0.8
    else:
        return 0.65 * (np.max([cloudn,0])) + 0.15



def calc_directr(diffuser):
    """ Calculate direct (incident) fraction of total incoming shortwave radiation
    Following B+A eqn. 2
    """
    return 1 - diffuser



def calc_Qn(directr, inswrad, solaltr):
    """ Calculate radiation received at surface normal to sun's rays 
    B+A eqn. 4
    """
    return directr * inswrad / np.sin(solaltr)



def calc_Qi(Qn, solaltr, slope, solaz, aspect):
    """ Calculate direct (incident) component of shortwave radiation
    B+A eqn. 5
    """
    return Qn * (((np.sin(solaltr))*(np.cos((slope)*(math.pi/180))))+((np.cos(solaltr))*(np.sin((slope)*(math.pi/180)))*(np.cos((solaz)-((aspect)*(math.pi/180))))))



def calc_Q_Wm2(albedo, Qi, diffuser, inswrad, slope):
    """ Calculate net shortwave radiation (W m-2)
    B+A eqn. 7
    """
    return ((1-albedo)*Qi)+((1-albedo)*diffuser*((inswrad*((np.cos(((slope)*(math.pi/180))/2))**2))+(albedo*((np.sin(((slope)*(math.pi/180))/2))**2))))



def calc_Q_melt(Q_Wm2):
    """ Calculate melting by (net) shortwave radiation (W m-2) """
    return Q_Wm2 * 0.0107784



def calc_eo(airtemp, elevation, met_elevation, lapse):
    """ Calculate clear sky emissivity 
    B+A see after eqn. 10
    """
    return 8.733 * (0.001 * (((airtemp-(lapse*(elevation-met_elevation))) + 273.16)**0.788))



def calc_e_star(cloudn, eo):
    """ Calculate sky effective emissivity 
    B+A eqn. 10
    """
    return (1+(0.26*(np.max([cloudn,0]))))* eo



def calc_lwin(e_star, airtemp, lapse, elevation, met_elevation):
    """ Calculate downwelling longwave radiation (W m-2)
    B+A eqn. 9 
    """
    return e_star * (5.7*(10**-8))*(((airtemp-(lapse*(elevation-met_elevation)))+273.16)**4)



def calc_lnet_Wm2(lwin, lwout=316.):
    """ Calculate net longwave radiation (W m-2)
    B+A eqn. 8 
    """
    return lwin - lwout



def calc_lnet_melt(lnet_Wm2):
    """ Calculate melt due to (net) longwave radiation """
    return lnet_Wm2 * 0.0107784



def calc_atmpres(elevation):
    """ Calculate atmospheric pressure """
    return 100000*(1-(elevation*0.0001))



def calc_de(avp, elevation, met_elevation, lapse):
    return (avp-(45*((elevation-met_elevation)*lapse)))-610.8



def calc_spehum(avp, atmpres):
    """ Calculate specific humidity of air """
    return avp / atmpres 



def calc_spehtair(spehum):
    """ Calculate specific heat of air at constant pressure (J kg-1 K-1)
    B+A para following eqn. 12
    """
    return 1004.67*(1+(0.84*spehum))



def calc_Re_star(U_star, roughness):
    """ Estimate scaling length for roughness
    B+A para following eqn. 12 and 16
    """
    return (U_star*roughness)/0.00001461



def calc_ln_zt(roughness, Re_star):
    """ Estimate scaling length for temperature
    B+A para following eqn. 12 and 16
    """
    return np.log(roughness)+0.317-(0.565*(np.log(Re_star)))-(0.183*((np.log(Re_star))**2))



def calc_ln_ze(roughness, Re_star):
    """ Estimate scaling length for humidity 
    B+A para following eqn. 12 and 16
    """
    return np.log(roughness)+0.396-(0.512*np.log(Re_star))-(0.18*((np.log(Re_star))**2))



def calc_L(U_star, airtemp, lapse, elevation, met_elevation, spehtair, Qs_Wm2):
    """ Estimate Monin-Obukhov length scale 
    B+A eqn. 13
    """
    return (1.225*((U_star)**3)*((((airtemp-(lapse*(elevation-met_elevation)))+273.16)+273.16)/2)*(spehtair))/(0.4*9.81*Qs_Wm2)



def calc_U_star_z_L(windspd, roughness):
    """ Initial estimate of friction velocity
    B+A para following eqn. 14
    """
    return (0.4*windspd)/((np.log(2/roughness)))



def calc_U_star_full(windspd, roughness, L):
    """ Full estimate of friction velocity
    B+A eqn. 14
    """
    return (0.4*windspd)/((np.log(2/roughness))+(5*(2/L)))



def calc_Qsz_L0(spehtair, windspd, airtemp, lapse, elevation, met_elevation, roughness, ln_zt):
    """ Initial estimate of sensible heat flux (without L)
    B+A para following eqn. 14
    """
    return (1.225*spehtair*0.16*windspd*(airtemp-(lapse*(elevation-met_elevation))))/(((np.log(2/roughness)))*(((np.log(2))-ln_zt)))



def calc_Qlz_L0(windspd, de, atmpres, roughness, ln_ze):
    """ Initial estimate for latent heat flux (without L)
    B+A para following eqn. 14
    """
    return (1.225*0.622*2500000*0.16*windspd*(de/atmpres))/((np.log(2/roughness))*((np.log(2))-(ln_ze)))



def calc_Qs_full(spehtair, windspd, airtemp, lapse, elevation, met_elevation, roughness, L, ln_zt):
    """ Full estimate of Sensible Heat Flux W m-2 
    B+A eqn. 11
    """
    return (1.225*spehtair*0.16*windspd*(airtemp-(lapse*(elevation-met_elevation))))/(((np.log(2/roughness))+(5*(2/L)))*(((np.log(2))-ln_zt)+(5*(2/L))))



def calc_QI_full(windspd, de, atmpres, roughness, ln_ze, L):
    """ Full estimate of Latent Heat Flux W m-2 
    B+A eqn. 12
    """
    return (1.225*0.622*2500000*0.16*windspd*(de/atmpres))/(((np.log(2/roughness))+(5*(2/L)))*(((np.log(2))-(ln_ze))+(5*(2/L))))



def calc_turbulent_fluxes(windspd, roughness, spehtair, airtemp, de, atmpres,
    lapse, elevation, met_elevation,
    max_n=25, tol=0.001,
    verbose=False, return_steps=False):
    """ Iteratively solve the turbulent fluxes and the Monin-Obukhov scale
    B+A page 653.
    Inputs:
    n : maximum number of iterations
    tol : tolerance threshold for iteration
    verbose : if True then print information about iterations
    return_steps : if True then return values of intermediary calculations
        as per Brock and Arnold (2000)
    Returns:
    tuple (SHF, LHF) in units W m-2
    if return_steps=True then returns:
    tuple (U_star, Re_star, ln_zt, ln_ze, Qs_Wm2, QI_Wm2, L)
    """

    # Solve initial conditions
    U_star = calc_U_star_z_L(windspd, roughness)
    Re_star = calc_Re_star(U_star, roughness)

    if verbose:
        print('Initial conditions: U_star: %s, Re_star: %s' %(U_star, Re_star))

    if verbose:
        print('Iteration block 1 . . . ')
        print('ln_zt  ln_ze  Qs_Wm2  Ql_Wm2')
    n = 0
    while n < max_n:
        ln_zt = calc_ln_zt(roughness, Re_star)
        ln_ze = calc_ln_ze(roughness, Re_star)
        Qs_Wm2 = calc_Qsz_L0(spehtair, windspd, airtemp, lapse, elevation, 
            met_elevation, roughness, ln_zt)
        Ql_Wm2 = calc_Qlz_L0(windspd, de, atmpres, roughness, ln_ze)

        if verbose:
            print (ln_zt, ln_ze, Qs_Wm2, Ql_Wm2)

        n += 1

    if verbose:
        print('Iteration block 2 . . . ')
        print('U_star  Re_star  ln_zt  ln_ze  Qs_Wm2  QI_Wm2  L')
    n = 0
    L = 0.
    L_old = 1.
    while (n < max_n) and (np.abs(L - L_old) > tol):
        L_old = L
        L = calc_L(U_star, airtemp, lapse, elevation, met_elevation, spehtair, Qs_Wm2)
        Re_star = calc_Re_star(U_star, roughness)
        ln_zt = calc_ln_zt(roughness, Re_star)
        ln_ze = calc_ln_ze(roughness, Re_star)
        Qs_Wm2 = calc_Qs_full(spehtair, windspd, airtemp, lapse, elevation, met_elevation, roughness, L, ln_zt)
        QI_Wm2 = calc_QI_full(windspd, de, atmpres, roughness, ln_ze, L)
        U_star = calc_U_star_full(windspd, roughness, L)

        if verbose:
            print(U_star, Re_star, ln_zt, ln_ze, Qs_Wm2, QI_Wm2, L)

        n += 1

    if return_steps:
        return (U_star, Re_star, ln_zt, ln_ze, Qs_Wm2, QI_Wm2, L)
    else:
        return (Qs_Wm2, QI_Wm2)



def calc_shf_melt(windspd, Qs_Wm2, airtemp):
    """ Calculate melting (mm w.e.) by SHF """
    if windspd > 2:
        return Qs_Wm2 * 0.0107784
    else:
        if windspd/airtemp < 0.3:
            return 0
        else:
            if (airtemp < 1.5) and (airtemp > -1.5) and (windspd < 1.5):
                return 0
            else:
                if (airtemp < 2) and (airtemp > -2) and (windspd < 1):
                    return 0
                else:
                    return Qs_Wm2 * 0.0107784



def calc_lhf_melt(windspd, QI_Wm2, airtemp):
    """ Calculate melting (mm w.e.) by LHF """
    if windspd > 2:
        return QI_Wm2 * 0.0107784
    else:
        if (windspd / airtemp) < 0.3:
            return 0
        else:
            if (airtemp < 1.5) and (airtemp > -1.5) and (windspd < 1.5):
                return 0
            else:
                if (airtemp < 2) and (airtemp > -2) and (windspd < 1):
                    return 0
                else:
                    return QI_Wm2 * 0.0107784



def calc_melt_total(SWR, LWR, SHF, LHF):
    """ Sum melt components (mm w.e.) """
    melt_total = SWR + LWR + SHF + LHF
    if melt_total < 0:
        return 0.
    else:
        return melt_total



def calculate_seb(lat, lon, lon_ref, day, time, summertime,
    slope, aspect, elevation, met_elevation, lapse,
    inswrad, avp, airtemp, windspd, albedo, roughness):
    """ Convenience function to solve energy balance for a single timestep.
    Returns:
        (SWR, LWR, SHF, LHF)
    """

    ## Solar characteristics/quantities
    dayang = calc_dayang(day)
    eqtim = calc_eqtim(dayang)
    solhour = calc_solhour(time, lon, lon_ref, eqtim, summertime)
    soldec = calc_soldec(day)
    solhran = calc_solhran(solhour)
    solaltr = calc_solaltr(lat, soldec, solhran)
    solaltd = calc_solaltd(solaltr)
    cossolaz = calc_cossolaz(lat, solaltr, soldec)
    sinolaz = calc_sinolaz(soldec, solhran, solaltr)
    solaz = calc_solaz(sinolaz, cossolaz)
    solaz360 = calc_solaz360(solaz)

    ## Calculate shortwave radiation
    cloudn = calc_cloudn(inswrad, elevation, solaltr)
    diffuser = calc_diffuser(cloudn)
    directr = calc_directr(diffuser)
    Qn = calc_Qn(directr, inswrad, solaltr)
    Qi = calc_Qi(Qn, solaltr, slope, solaz, aspect)
    # Net shortwave radiation
    Q_Wm2 = calc_Q_Wm2(albedo, Qi, diffuser, inswrad, slope)
    
    ## Calculate longwave radiation
    eo = calc_eo(airtemp, elevation, met_elevation, lapse)
    e_star = calc_e_star(cloudn, eo)
    lwin = calc_lwin(e_star, airtemp, lapse, elevation, met_elevation)
    # Net longwave radiation
    lnet_Wm2 = calc_lnet_Wm2(lwin)

    ## Calculate turbulent fluxes
    atmpres = calc_atmpres(elevation)
    de = calc_de(avp, elevation, met_elevation, lapse)
    spehum = calc_spehum(avp, atmpres)
    spehtair = calc_spehtair(spehum)
    # Return individual sensible and latent heat fluxes
    Qs_Wm2, QI_Wm2 = calc_turbulent_fluxes(windspd, roughness, spehtair, 
        airtemp, de, atmpres, lapse, elevation, met_elevation)

    return (Q_Wm2, lnet_Wm2, Qs_Wm2, QI_Wm2)



def calculate_melt(swnet, lwnet, shf, lhf, windspd, airtemp):
    """ Convert to melt quantities """
    swnet_melt = calc_Q_melt(swnet) 
    lwnet_melt = calc_lnet_melt(lwnet)
    SHF_melt = calc_shf_melt(windspd, shf, airtemp)
    LHF_melt = calc_lhf_melt(windspd, lhf, airtemp)
    melt_total = calc_melt_total(swnet_melt, lwnet_melt, SHF_melt, LHF_melt)

    return (swnet_melt, lwnet_melt, SHF_melt, LHF_melt, melt_total)









"""
Not needed - these 'columns' are replicas of Qsfull and QIfull
def calc_Qs_Wm2(spehtair, winspd, airtemp, lapse, elevation, met_elevation, roughness, L, ln_zt):
    return (1.225*spehtair*0.16*windspd*(airtemp-(lapse*(elevation-met_elevation))))/(((np.log(2/roughness))+(5*(2/L)))*(((np.log(2))-ln_zt)+(5*(2/L))))
def calc_QI_Wm2(windspd, de, atmpres, roughness, L, ln_ze):
    return (1.225*0.622*2500000*0.16*windspd*(de/atmpres))/(((np.log(2/roughness))+(5*(2/L)))*(((np.log(2))-(ln_ze))+(5*(2/L))))
"""