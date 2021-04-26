# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 00:14:29 2021
@author: elohe
"""
from statistics import mean
import math
import pandas as pd
import numpy as np
import sys
import csv
import Cread_bf_file
import get_scaled_csi
from itertools import permutations
from numpy import inf

from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tkinter import re #operaciones de matching de expresiones regulares

"""
def db(x):
    decibels = 10*math.log10
    return decibels
def dbinv(x):
    ret =  10**(x/10)
    return ret
def get_total_rss(csi_st):
    ## Condición de error puede manejarse con try - catch
    try:
        rssi_mag = 0
        
        if (csi_st.rssi_a != 0):
            rssi_mag = rssi_mag + dbinv(csi_st.rssi_a)
            
        if (csi_st.rssi_b != 0):
            rssi_mag = rssi_mag + dbinv(csi_st.rssi_b)
        
        if (csi_st.rssi_c != 0):
            rssi_mag = rssi_mag + dbinv(csi_st.rssi_c)
    
        ret = (db(rssi_mag)) - 44 - csi_st.agc
        return ret
    except:
        print("Se ha producido un error en 'get_total_rss'", sys.exc_info()[0])
        return
        
def get_scaled_csi(csi_st):    
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
"""
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


# Datfile_convert script
#Se abre una ventana de dialogo para solicitar el archivo csv
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
file_path = askopenfilenames(parent=root, title='Choose a file', initialdir='datos_crudos',
                               filetypes=(("DAT Files", "*.dat"),))

csv_headers = "csi_headers.csv"
csv_cols = pd.read_csv(csv_headers)
csv_col_list = csv_cols["Column_Names"].tolist()

for i in range(len(file_path)):
    splitted = file_path[i].split("/")
    file_name = splitted[-1]
    folder_name = file_path[i].replace(file_name,'') 
    
    csi_trace = Cread_bf_file.read_bf_file(file_path[i])
    #csi_columns = ["timestamp_low", "bfee_count", "Nrx", "Ntx", "rssi_a", "rssi_b", "rssi_c",
    #            "noise", "agc", "perm", "fake_rate_n_flags", "csi"]
    #csi_frame = pd.DataFrame(csi_trace, columns=csi_columns)
    
    # eliminate empty cell
    #xx = find(cellfun('isempty', csi_trace))
    ###csi_trace[csi_trace == ""] = []

    #csi_trace[xx] = []

    # Extract CSI information for each packet
    print('Have CSI for {} packets\n'.format(len(csi_trace)))

    # Scaled into linear
    csi = np.zeros([len(csi_trace),3,30])
    timestamp = np.zeros([1, len(csi_trace)])
    temp = []
    
    for packet_index in range(len(csi_trace)):
        csi[packet_index,:,:] = get_scaled_csi.get_scaled_csi(csi_trace[packet_index])
        timestamp[0][packet_index] = csi_trace[packet_index][0] * 1.0e-6
    
    timestamp = np.transpose(timestamp)
    
    csi_amp_matrix = np.zeros([len(csi_trace), 90])
    csi_amp_matrix[:, 0:30] = get_scaled_csi.db(abs(np.squeeze(csi[:, 0, :])))
    csi_amp_matrix[:, 30:60] = get_scaled_csi.db(abs(np.squeeze(csi[:, 1, :])))
    csi_amp_matrix[:, 60:90] = get_scaled_csi.db(abs(np.squeeze(csi[:, 2, :])))
    
    """                  
    # File export                 
    csi_amp_matrix = np.transpose(get_scaled_csi.db(abs(np.squeeze(csi))), (2, 3, 1))
    csi_phase_matrix = np.transpose(np.angle(np.squeeze(csi)), (2, 3, 1))
    
    csi_phase_matrix2 = []
    for k in range(len(np.size(csi_phase_matrix,1))):
        for j in range (len(np.size(csi_phase_matrix,3))):
            csi_phase_matrix2[k,:,j] = np.transpose(phase_calibration(csi_phase_matrix[k,:,j]))
        
    for packet_index in range(len(csi_trace)):
        temp = [temp, np.hstack(np.reshape(np.transpose(csi_amp_matrix[:,:,packet_index]),[0,89]), np.reshape(np.transpose(csi_phase_matrix2[:,:,packet_index]),[0,89]))]
    
    """
    csi_amp_matrix[csi_amp_matrix == -inf] = np.nan  #-Inf values replaced by NaN. By: Emmanuel L
    
    
    FileName_new = file_name.replace(".dat", ".csv")
    #csvwrite([PathName + FileName_new + '.csv'],
    
    CsvNewFile = np.zeros([len(csi_trace), len(np.transpose(csi_amp_matrix)) + 1])
             
    CsvNewFile = np.hstack([timestamp, csi_amp_matrix])
    DFCsv = pd.DataFrame(CsvNewFile, columns=csv_col_list)
    DFCsv.to_csv(r''+ folder_name + FileName_new, index = False)