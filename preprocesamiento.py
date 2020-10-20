# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:40:41 2020

Script para preprocesar los datos crudos, se aplican las operaciones de:
    Imputación de Datos, Eliminación de Ruido, y Normalización.

@author: elohe
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk 
from tkinter.filedialog import askopenfilename
from tkinter import re #operaciones de matching de expresiones regulares
from DataBaseConnection import DataBase
from ClasesNumericas import ClasesNum
import datetime

##Función de normalización manual
def normalizar(df):
    result= df.copy()
    for dato in df.columns:
        max_value = df[dato].max()
        min_value = df[dato].min()
        
        result[dato] = (df[dato] - min_value) / (max_value - min_value)
    return result

##Función de eliminación de ruido (Por vector)
def ruido(i):
    mirror_i = np.transpose(i)
    w = 3
    ven = w
    
    prev = mirror_i[0:ven-1]
    prev_mean = np.mean(prev)
    
    modified_length = len(mirror_i)-2
    z_table = []
    
    ##Para length del primer valor hasta el final del vector
    for data in range(modified_length):
        y = mirror_i
        if ven <= len(mirror_i):
            z = y[data:ven]
            ven = ven +1
        
        acumulado = np.mean(z)
        z_table.append(acumulado)
    
    penultimo = mirror_i[-2]
    ultimo = mirror_i[-1]
    last = [penultimo, ultimo]
    last_mean = np.mean(last)
    
    z_table.insert(0, prev_mean)
    z_table.append(last_mean)
    
    return z_table
    
    

#Se abre una ventana de dialogo para solicitar el archivo csv
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
file_path = askopenfilename() #Se abre el explorador de archivos y se guarda la selección
splitted = file_path.split("/")
file_name = splitted[-1]
folder_name = file_path.replace(file_name,'') 

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

#Eliminación de ruido
trn_matrix = trn_NaN_2_0.values
rows_matrix = len(trn_matrix)
cols_matrix = len(np.transpose(trn_matrix))

trn_sin_ruido = trn_matrix[:, 0:cols_matrix] #Sin el timestamp, 180 variables
trn_sin_ruido_collected = trn_sin_ruido * 0

for dat in range(cols_matrix):
    trn_sin_ruido_collected[:, dat] = ruido(trn_sin_ruido[:, dat])

sin_ruido_df = pd.DataFrame(trn_sin_ruido_collected, columns=csv_col_list)

minimos = np.zeros((1, 90))
maximos = np.zeros((1, 90))
trn_nruido_ntime = trn_sin_ruido_collected[:, 1:-1]
tiempo = datetime.datetime.now()

for col in range(len(np.transpose(minimos))):
    minimos[:, col] = trn_nruido_ntime[:, col].min()
    maximos[:, col] = trn_nruido_ntime[:, col].max()

"""

Se deben almacenar datos mínimos y máximos de cada subcarrier, de los datos originales 
en la tabla de MIN_MAX de la base de datos, por cada movimiento.


"""
### Mínimos
minimos_str = []
mov_identifier = file_name.split("_")[-4]
tipo_movimiento = ClasesNum(mov_identifier)
clase = tipo_movimiento.val_int_clase
id_min = 0

for col in range(len(np.transpose(minimos))):
     minimos_str.append(str(minimos[0, col]))
     
minimos_str.append("MINIMO")
minimos_str.append(str(tiempo))
minimos_str.append(str(clase))

database = DataBase()
sql_minimos = "INSERT INTO min_max(ANTENA_1_AMP_SUB1, ANTENA_1_AMP_SUB2, ANTENA_1_AMP_SUB3, ANTENA_1_AMP_SUB4, ANTENA_1_AMP_SUB5, ANTENA_1_AMP_SUB6, ANTENA_1_AMP_SUB7, ANTENA_1_AMP_SUB8, ANTENA_1_AMP_SUB9, ANTENA_1_AMP_SUB10, \
ANTENA_1_AMP_SUB11, ANTENA_1_AMP_SUB12, ANTENA_1_AMP_SUB13, ANTENA_1_AMP_SUB14, ANTENA_1_AMP_SUB15, ANTENA_1_AMP_SUB16, ANTENA_1_AMP_SUB17, ANTENA_1_AMP_SUB18, ANTENA_1_AMP_SUB19, ANTENA_1_AMP_SUB20, \
ANTENA_1_AMP_SUB21, ANTENA_1_AMP_SUB22, ANTENA_1_AMP_SUB23, ANTENA_1_AMP_SUB24, ANTENA_1_AMP_SUB25, ANTENA_1_AMP_SUB26, ANTENA_1_AMP_SUB27, ANTENA_1_AMP_SUB28, ANTENA_1_AMP_SUB29, ANTENA_1_AMP_SUB30, \
ANTENA_2_AMP_SUB1, ANTENA_2_AMP_SUB2, ANTENA_2_AMP_SUB3, ANTENA_2_AMP_SUB4, ANTENA_2_AMP_SUB5, ANTENA_2_AMP_SUB6, ANTENA_2_AMP_SUB7, ANTENA_2_AMP_SUB8, ANTENA_2_AMP_SUB9, ANTENA_2_AMP_SUB10, \
ANTENA_2_AMP_SUB11, ANTENA_2_AMP_SUB12, ANTENA_2_AMP_SUB13, ANTENA_2_AMP_SUB14, ANTENA_2_AMP_SUB15, ANTENA_2_AMP_SUB16, ANTENA_2_AMP_SUB17, ANTENA_2_AMP_SUB18, ANTENA_2_AMP_SUB19, ANTENA_2_AMP_SUB20, \
ANTENA_2_AMP_SUB21, ANTENA_2_AMP_SUB22, ANTENA_2_AMP_SUB23, ANTENA_2_AMP_SUB24, ANTENA_2_AMP_SUB25, ANTENA_2_AMP_SUB26, ANTENA_2_AMP_SUB27, ANTENA_2_AMP_SUB28, ANTENA_2_AMP_SUB29, ANTENA_2_AMP_SUB30, \
ANTENA_3_AMP_SUB1, ANTENA_3_AMP_SUB2, ANTENA_3_AMP_SUB3, ANTENA_3_AMP_SUB4, ANTENA_3_AMP_SUB5, ANTENA_3_AMP_SUB6, ANTENA_3_AMP_SUB7, ANTENA_3_AMP_SUB8, ANTENA_3_AMP_SUB9, ANTENA_3_AMP_SUB10, \
ANTENA_3_AMP_SUB11, ANTENA_3_AMP_SUB12, ANTENA_3_AMP_SUB13, ANTENA_3_AMP_SUB14, ANTENA_3_AMP_SUB15, ANTENA_3_AMP_SUB16, ANTENA_3_AMP_SUB17, ANTENA_3_AMP_SUB18, ANTENA_3_AMP_SUB19, ANTENA_3_AMP_SUB20, \
ANTENA_3_AMP_SUB21, ANTENA_3_AMP_SUB22, ANTENA_3_AMP_SUB23, ANTENA_3_AMP_SUB24, ANTENA_3_AMP_SUB25, ANTENA_3_AMP_SUB26, ANTENA_3_AMP_SUB27, ANTENA_3_AMP_SUB28, ANTENA_3_AMP_SUB29, ANTENA_3_AMP_SUB30, VALOR, DATE_CREATED, FK_MOVIMIENTO)  \
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

        
try:
   database.cursor.execute(sql_minimos, minimos_str)
   id_min = database.connection.insert_id()
   database.connection.commit()
   #database.close()
    
except Exception as e:
    raise
    
##Máximos
maximos_str = []

for col in range(len(np.transpose(maximos))):
     maximos_str.append(str(maximos[0, col]))
     
maximos_str.append("MAXIMO")
maximos_str.append(str(tiempo))
maximos_str.append(str(clase))
maximos_str.append(str(id_min))

database = DataBase()
sql_maximos = "INSERT INTO min_max(ANTENA_1_AMP_SUB1, ANTENA_1_AMP_SUB2, ANTENA_1_AMP_SUB3, ANTENA_1_AMP_SUB4, ANTENA_1_AMP_SUB5, ANTENA_1_AMP_SUB6, ANTENA_1_AMP_SUB7, ANTENA_1_AMP_SUB8, ANTENA_1_AMP_SUB9, ANTENA_1_AMP_SUB10, \
ANTENA_1_AMP_SUB11, ANTENA_1_AMP_SUB12, ANTENA_1_AMP_SUB13, ANTENA_1_AMP_SUB14, ANTENA_1_AMP_SUB15, ANTENA_1_AMP_SUB16, ANTENA_1_AMP_SUB17, ANTENA_1_AMP_SUB18, ANTENA_1_AMP_SUB19, ANTENA_1_AMP_SUB20, \
ANTENA_1_AMP_SUB21, ANTENA_1_AMP_SUB22, ANTENA_1_AMP_SUB23, ANTENA_1_AMP_SUB24, ANTENA_1_AMP_SUB25, ANTENA_1_AMP_SUB26, ANTENA_1_AMP_SUB27, ANTENA_1_AMP_SUB28, ANTENA_1_AMP_SUB29, ANTENA_1_AMP_SUB30, \
ANTENA_2_AMP_SUB1, ANTENA_2_AMP_SUB2, ANTENA_2_AMP_SUB3, ANTENA_2_AMP_SUB4, ANTENA_2_AMP_SUB5, ANTENA_2_AMP_SUB6, ANTENA_2_AMP_SUB7, ANTENA_2_AMP_SUB8, ANTENA_2_AMP_SUB9, ANTENA_2_AMP_SUB10, \
ANTENA_2_AMP_SUB11, ANTENA_2_AMP_SUB12, ANTENA_2_AMP_SUB13, ANTENA_2_AMP_SUB14, ANTENA_2_AMP_SUB15, ANTENA_2_AMP_SUB16, ANTENA_2_AMP_SUB17, ANTENA_2_AMP_SUB18, ANTENA_2_AMP_SUB19, ANTENA_2_AMP_SUB20, \
ANTENA_2_AMP_SUB21, ANTENA_2_AMP_SUB22, ANTENA_2_AMP_SUB23, ANTENA_2_AMP_SUB24, ANTENA_2_AMP_SUB25, ANTENA_2_AMP_SUB26, ANTENA_2_AMP_SUB27, ANTENA_2_AMP_SUB28, ANTENA_2_AMP_SUB29, ANTENA_2_AMP_SUB30, \
ANTENA_3_AMP_SUB1, ANTENA_3_AMP_SUB2, ANTENA_3_AMP_SUB3, ANTENA_3_AMP_SUB4, ANTENA_3_AMP_SUB5, ANTENA_3_AMP_SUB6, ANTENA_3_AMP_SUB7, ANTENA_3_AMP_SUB8, ANTENA_3_AMP_SUB9, ANTENA_3_AMP_SUB10, \
ANTENA_3_AMP_SUB11, ANTENA_3_AMP_SUB12, ANTENA_3_AMP_SUB13, ANTENA_3_AMP_SUB14, ANTENA_3_AMP_SUB15, ANTENA_3_AMP_SUB16, ANTENA_3_AMP_SUB17, ANTENA_3_AMP_SUB18, ANTENA_3_AMP_SUB19, ANTENA_3_AMP_SUB20, \
ANTENA_3_AMP_SUB21, ANTENA_3_AMP_SUB22, ANTENA_3_AMP_SUB23, ANTENA_3_AMP_SUB24, ANTENA_3_AMP_SUB25, ANTENA_3_AMP_SUB26, ANTENA_3_AMP_SUB27, ANTENA_3_AMP_SUB28, ANTENA_3_AMP_SUB29, ANTENA_3_AMP_SUB30, VALOR, DATE_CREATED, FK_MOVIMIENTO, ASOCIADO)  \
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"



try:
   database.cursor.execute(sql_maximos, maximos_str)
   database.connection.commit()
   database.close()
    
except Exception as e:
    raise

#Se normalizan los datos usados para pruebas. 
#Los datos nuevos se normalizarán con la tabla MIN_MAX

##trn_normalizado= normalizar(sin_ruido_df)
##trn_normalizado['timestamp'] = tim_normalizado['timestamp']


#Saber interpretar el nombre en automático
##trn_normalizado.to_csv(r''+ 'preprocesados' +'\input_' + file_name, index = False, header=True)