# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:40:41 2020

@author: elohe
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk 
from tkinter.filedialog import askopenfilename
from tkinter import re #operaciones de matching de expresiones regulares

##Función de normalización manual
def normalizar(df):
    result= df.copy()
    for dato in df.columns:
        max_value = df[dato].max()
        min_value = df[dato].min()
        
        result[dato] = (df[dato] - min_value) / (max_value - min_value)
    return result

#Se abre una ventana de dialogo para solicitar el archivo csv
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
file_path = askopenfilename() #Se abre el explorador de archivos y se guarda la selección

#Para guardar el directorio donde se encontraba el archivo seleccionado
match = re.search(r'/.*\..+', file_path) #matches name of file
file_position = file_path.find(match.group()) #defines position of filename in file path

save_path = file_path[0: file_position+1] #extracts the saving path.

#Se añaden los encabezados
csv_headers = "csi_headers.csv"
csv_cols = pd.read_csv(csv_headers)
csv_col_list = csv_cols["Column_Names"].tolist()

trn = pd.read_csv(file_path, names=csv_col_list)
trn_len = len(trn)
trn_tim = trn['timestamp']

#Se convierte en dataframe
trn_tim_df = pd.DataFrame(trn_tim)
#Se calculan los segundos de actividad
tim_normalizado = normalizar(trn_tim_df) * (trn_tim.max() - trn_tim.min())

#Imputación de datos
inf_estadistica_trn = trn.describe() #Por lo tanto existen datos faltantes.

#Razones por las que faltan datos:
#1. Se puede optimizar el procesa de extracción de los datos
#2. Debido a la recolección de los datos

#Se pone nan como 0
trn_NaN_2_0 = trn.fillna(1.0000e-5)

#Se normalizan los datos
trn_normalizado= normalizar(trn_NaN_2_0)
trn_normalizado["timestamp"] = tim_normalizado

#Eliminación de ruido

