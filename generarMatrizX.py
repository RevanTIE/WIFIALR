# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:21:14 2020

@author: elohe

Script que toma 6 columnas de cada matriz de PCA, de cada movimiento,
y construye una nueva matriz, que será dividida en training, testing y
validation.

Si se opta por la segunda metodología, crear copia de este script para generar 
X y Y basados en los datos sin PCA (datos preprocesados).

"""

import pandas as pd
import numpy as np
import os
import glob
from ClasesNumericas import ClasesNum

ruta = 'pca'
os.chdir(ruta)
pca_list = glob.glob('*.{}'.format('csv'))


X = np.zeros((139776, 7))
Y = np.zeros((19968, 1)) #Clases

inc_rows_start = 0
inc_rows_end = 19968


"""
Poner condición, si se considera la totalidad de los datos de todos los 
movimientos para fines de training-test; o un dataset nuevo para validación.
"""
for mov in range(len(pca_list)):
    matrix = pd.read_csv(pca_list[mov])
    pca_matrix = matrix.iloc[0:19968, 0:6]
    
    movimiento = pca_list[mov].split("_")[-4]
    tipo_movimiento = ClasesNum(movimiento).val_int_clase
    
    Y[Y==mov] = tipo_movimiento
    
    X[inc_rows_start:inc_rows_end, 0] = Y[:, 0]
    X[inc_rows_start:inc_rows_end, 1:7] = pca_matrix
    
    #8 if (inc_cols_start == 0) else 7
    inc_rows_start = inc_rows_start + 19968
    inc_rows_end = inc_rows_end + 19968


df_X = pd.DataFrame(X[:, 1:len(X.transpose())])
df_Y = pd.DataFrame(X[:, 0])

os.chdir('../trn_tst')

df_X.to_csv(r''+ 'X.csv', index = False, header=False)
df_Y.to_csv(r''+ 'Y.csv', index = False, header=False)