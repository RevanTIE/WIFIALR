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
from ClasesNumericas import ClasesNum
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tkinter.filedialog import askopenfilename

root = Tk()  # Elimina la ventana de Tkinter
root.withdraw()  # Ahora se cierra

"""
Poner condición, si se considera la totalidad de los datos de todos los 
movimientos para fines de training-test; o un dataset nuevo para validación.
"""

consideracion = input("¿Desea tratar los datos como Matriz de Training/Test? S = SI, N = NO: ")

if (consideracion == "S"):
    pca_list = askopenfilenames(parent=root, title='Choose a file', initialdir='pca',
                                filetypes=(("CSV Files", "*.csv"),))
    X_list = []

    for mov in range(len(pca_list)):
        X_matrix = pd.read_csv(pca_list[mov])
        X_list.append(len(X_matrix))

    suma_x_list = sum(X_list)
    X = np.zeros((suma_x_list, 7))

    inc_rows_start = 0
    inc_rows_end = 0

    for mov in range(len(pca_list)):
        splitted = pca_list[mov].split("/")
        file_name = splitted[-1]
        folder_name = pca_list[mov].replace(file_name, '')

        matrix = pd.read_csv(pca_list[mov])
        count_rows = len(matrix)

        if (mov == 0):
            inc_rows_end = X_list[mov]

        Y = np.zeros((count_rows, 1))
        pca_matrix = matrix.iloc[:, 0:6]

        movimiento = pca_list[mov].split("_")[-4]
        tipo_movimiento = ClasesNum(movimiento).val_int_clase

        Y[Y == 0] = tipo_movimiento

        X[inc_rows_start:inc_rows_end, 0] = Y[:, 0]
        X[inc_rows_start:inc_rows_end, 1:7] = pca_matrix

        # 8 if (inc_cols_start == 0) else 7
        if (mov != len(pca_list) - 1):
            inc_rows_start = inc_rows_end
            inc_rows_end += X_list[mov + 1]

    df_X = pd.DataFrame(X[:, 1:len(X.transpose())])
    df_Y = pd.DataFrame(X[:, 0])

    df_X.to_csv(r'' + 'trn_tst' + '/X.csv', index=False, header=False)
    df_Y.to_csv(r'' + 'trn_tst' + '/Y.csv', index=False, header=False)

else:
    pca_file = askopenfilename(parent=root, title='Choose a file', initialdir='pca',
                               filetypes=(("CSV Files", "*.csv"),))
    splitted = pca_file.split("/")
    file_name = splitted[-1]

    matrix = pd.read_csv(pca_file)
    pca_matrix = matrix.iloc[:, 0:6]

    df_X = pd.DataFrame(pca_matrix)
    df_X.to_csv(r'' + 'datos_nuevos' + '/' + file_name, index=False, header=False)
