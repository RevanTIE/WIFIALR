# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:50:48 2021

@author: elohe

Uso de las librerías del Proyecto Gi-z/CSIKit 

"""

from CSIKit.reader import get_reader
from CSIKit.util import csitools
from CSIKit.util import csitools_frames_elopez
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#Se abre una ventana de dialogo para solicitar el archivo dat
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
file_path = askopenfilename(parent=root, title='Choose a file', initialdir='datos_crudos',
                               filetypes=(("DAT Files", "*.dat"),))


my_reader = get_reader(file_path)
csi_data = my_reader.read_file(file_path, scaled=True)
frames, csi_shape, no_subcarriers = csitools_frames_elopez.get_CSI_Frames(csi_data, metric="amplitude")

"""
csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
csi_matrix_inversa = csi_matrix.transpose()
timestamp_vector = csi_data.timestamps
"""