# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 00:14:29 2021

@author: elohe
"""
from statistics import mean
import math
import numpy as np

def dbinv(x):
    dbinv =  10**(x/10)
    return dbinv

def get_total_rss(csi_st):
    ## Condición de error puede manejarse con try - catch
    rssi_mag = 0
    
    if (csi_st.rssi_a != 0):
        rssi_mag = rssi_mag + dbinv(csi_st.rssi_a)
        
    if (csi_st.rssi_b != 0):
        rssi_mag = rssi_mag + dbinv(csi_st.rssi_b)
    
    if (csi_st.rssi_c != 0):
        rssi_mag = rssi_mag + dbinv(csi_st.rssi_c)

    ret = (10*math.log10(rssi_mag)) - 44 - csi_st.agc
    return ret

def phase_calibration(phasedata):
    calibrated_phase = []
    calibrated_phase2 = []
    
    calibrated_phase[0] = phasedata[0]
    difference = 0
    
    for i in range(1,30):
        temp = phasedata[i] - phasedata[i-1]
        
        if (abs(temp) > math.pi): 
            difference = difference + 1*np.sign[temp]
            
        calibrated_phase[i] = phasedata[i] - difference * 2 * math.pi
        
    k = (calibrated_phase[29] - calibrated_phase[0]) / (30 - 1)
    b = mean(calibrated_phase)
    
    for i in range(30):
        calibrated_phase2[i] = calibrated_phase[i] - k * i - b
    
    return calibrated_phase2

def __init__(self, csi_st):    
    csi = csi_st.csi
    csi_sq = csi * np.conj(csi)
    
    csi_pwr = np.sum(csi_sq[:]) ##Revisar si así se crea un vector de columna a partir de la matriz
    rssi_pwr = dbinv(get_total_rss(csi_st))
    scale = rssi_pwr / (csi_pwr / 30)
    
    if (csi_st.noise == -127):
        noise_db = -92
    else:
        noise_db = csi_st.noise
        
    thermal_noise_pwr = dbinv(noise_db)
    
    quant_error_pwr = scale * (csi_st.Nrx * csi_st.Ntx)
    
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr
    
    ret = csi * np.sqrt(scale / total_noise_pwr);
    if csi_st.Ntx == 2:
        ret = ret * np.sqrt(2)
    elif csi_st.Ntx == 3:
        ret = ret * np.sqrt(dbinv(4.5))
        
    return ret