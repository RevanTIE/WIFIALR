# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:21:14 2020

@author: elohe

Script que toma 6 columnas de cada matriz de PCA, de cada movimiento,
y construye una nueva matriz, que será dividida en training, testing y
validation.

"""

import pandas as pd
import numpy as np
import os
import glob

def clases_numericas(val_str_clase):
    val_int_clase = 0
    
    if (val_str_clase == 'BE'):
        val_int_clase = 0
        
    if (val_str_clase == 'FA'):
        val_int_clase = 1
    
    if (val_str_clase == 'PI'):
        val_int_clase = 2
    
    if (val_str_clase == 'RU'):
        val_int_clase = 3
    
    if (val_str_clase == 'SD'):
        val_int_clase = 4
    
    if (val_str_clase == 'SU'):
        val_int_clase = 5
    
    if (val_str_clase == 'WA'):
        val_int_clase = 6
    
    return val_int_clase

ruta = 'pca'
os.chdir(ruta)
pca_list = glob.glob('*.{}'.format('csv'))

X = np.zeros((42, 19969))
Y = []
inc_cols_start = 0
inc_cols_end = 6



for mov in range(len(pca_list)):
    matrix = pd.read_csv(pca_list[mov])
    pca_matrix = matrix.iloc[0:19968, 0:6]
    pca_matriz_tr = np.transpose(pca_matrix)
    
    Y.append(pca_list[mov].split("_")[-4])
    val_int_clase = clases_numericas(Y[mov])
    
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