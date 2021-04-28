# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:50:48 2021

@author: elohe

Uso de las librerías del Proyecto Gi-z/CSIKit 

"""

from CSIKit.reader import get_reader
import Amplitude
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import pandas as pd
from numpy import inf
import numpy as np

#Se abre una ventana de dialogo para solicitar el archivo dat
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
file_path = askopenfilenames(parent=root, title='Choose a file', initialdir='datos_crudos',
                               filetypes=(("DAT Files", "*.dat"),))

for i in range(len(file_path)):
    splitted = file_path[i].split("/")
    file_name = splitted[-1]
    folder_name = file_path[i].replace(file_name,'')

    my_reader = get_reader(file_path[i])
    csi_data = my_reader.read_file(file_path[i], scaled=True)
    (csi_1, csi_2, csi_3) = Amplitude.get_CSI_Frames(csi_data)

    csi_matrix_inversa_1 = csi_1.transpose()
    csi_matrix_inversa_2 = csi_2.transpose()
    csi_matrix_inversa_3 = csi_3.transpose()

    timestamp_vector = csi_data.timestamps

    csi_amp_matrix = np.zeros([csi_data.expected_frames, 90])
    csi_amp_matrix[:, 0:30] = csi_matrix_inversa_1
    csi_amp_matrix[:, 30:60] = csi_matrix_inversa_2
    csi_amp_matrix[:, 60:90] = csi_matrix_inversa_3

    csi_amp_matrix[csi_amp_matrix == -inf] = np.nan

    FileName_new = file_name.replace(".dat", ".csv")
    CsvNewFile = np.zeros([csi_data.expected_frames, len(np.transpose(csi_amp_matrix)) + 1])
    time_v = np.ravel(timestamp_vector)

    CsvNewFile =  np.c_[time_v, csi_amp_matrix]
    DFCsv = pd.DataFrame(CsvNewFile)
    DFCsv.to_csv(r'' + folder_name + FileName_new, index=False, header=False)
