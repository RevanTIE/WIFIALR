# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:50:48 2021

@author: elohe

Uso de las librerías del Proyecto Gi-z/CSIKit 

"""

from CSIKit.reader import get_reader
from CSIKit.util import csitools
import Amplitude
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import get_scaled_csi

from CSIKit.util.matlab import db
from typing import Tuple
import numpy as np

#Se abre una ventana de dialogo para solicitar el archivo dat
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
file_path = askopenfilename(parent=root, title='Choose a file', initialdir='datos_crudos',
                               filetypes=(("DAT Files", "*.dat"),))

splitted = file_path.split("/")
file_name = splitted[-1]
folder_name = file_path.replace(file_name,'')

my_reader = get_reader(file_path)
csi_data = my_reader.read_file(file_path, scaled=True)
(csi_1, csi_2, csi_3) = Amplitude.get_CSI_Frames(csi_data)

csi_matrix_inversa_1 = csi_1.transpose()
csi_matrix_inversa_2 = csi_2.transpose()
csi_matrix_inversa_3 = csi_3.transpose()

timestamp_vector = csi_data.timestamps

csi_amp_matrix = np.zeros([csi_data.expected_frames, 90])
csi_amp_matrix[:, 0:30] = csi_matrix_inversa_1
csi_amp_matrix[:, 30:60] = csi_matrix_inversa_2
csi_amp_matrix[:, 60:90] = csi_matrix_inversa_3

FileName_new = file_name.replace(".dat", ".csv")
CsvNewFile = np.zeros([csi_data.expected_frames, len(np.transpose(csi_amp_matrix)) + 1])
time_v = np.ravel(timestamp_vector)

CsvNewFile =  np.c_[time_v, csi_amp_matrix]
DFCsv = pd.DataFrame(CsvNewFile)
DFCsv.to_csv(r'' + folder_name + FileName_new, index=False, header=False)
