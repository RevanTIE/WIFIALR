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

X = np.zeros((42, 19969))
Y = []
inc_cols_start = 0
inc_cols_end = 6


"""
Poner condición, si se considera la totalidad de los datos de todos los 
movimientos para fines de training-test; o un dataset nuevo para validación.
"""
for mov in range(len(pca_list)):
    matrix = pd.read_csv(pca_list[mov])
    pca_matrix = matrix.iloc[0:19968, 0:6]
    pca_matriz_tr = np.transpose(pca_matrix)
    
    Y.append(pca_list[mov].split("_")[-4])
    tipo_movimiento = ClasesNum(Y[mov])
    val_int_clase = tipo_movimiento.val_int_clase
    #val_int_clase = clases_numericas(Y[mov])
    
    X[inc_cols_start:inc_cols_end, 0] = val_int_clase
    
    X[inc_cols_start:inc_cols_end, 1:19969] = pca_matriz_tr
    
    #8 if (inc_cols_start == 0) else 7
    
    inc_cols_start = inc_cols_start + 6
    inc_cols_end = inc_cols_end + 6


df_X = pd.DataFrame(X[:, 1:-1])
df_Y = pd.DataFrame(X[:, 0])

os.chdir('../trn_tst')
df_X_trans = np.transpose(df_X)
df_Y_trans = np.transpose(df_Y)

df_X_trans.to_csv(r''+ 'X.csv', index = False, header=False)
df_Y_trans.to_csv(r''+ 'Y.csv', index = False, header=False)