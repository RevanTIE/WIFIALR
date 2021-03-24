# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:21:14 2020

@author: elohe

Script que extrae el Dominio del Tiempo de cada matriz de PCA, de cada movimiento,
y construye una nueva matriz, que será dividida en training y testing.

"""

import pandas as pd
import numpy as np
from ClasesNumericas import ClasesNum
import tsfel
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tkinter.filedialog import askopenfilename

"""
    Atributos en el dominio del tiempo
"""
def AtribDomTiempo(df):
    # Retrieves a pre-defined feature configuration file to extract all available features
    cfg = tsfel.get_features_by_domain("statistical", "custom_features.json")

    # Extract features
    extracted_features = tsfel.time_series_features_extractor(cfg, df)
    return extracted_features

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
        X_vector = AtribDomTiempo(X_matrix)
        X_list.append(len(X_vector))

    suma_x_list = sum(X_list)
    valor = len(X_vector.transpose())
    X = np.zeros((suma_x_list, valor + 1))

    inc_rows_start = 0
    inc_rows_end = 0

    for mov in range(len(pca_list)):
        splitted = pca_list[mov].split("/")
        file_name = splitted[-1]
        folder_name = pca_list[mov].replace(file_name, '')

        matrix = pd.read_csv(pca_list[mov])
        vector = AtribDomTiempo(matrix)
        count_rows = 1 #len(vector)

        if (mov == 0):
            inc_rows_end = X_list[mov]

        Y = np.zeros((count_rows, 1))

        movimiento = pca_list[mov].split("_")[-4]
        tipo_movimiento = ClasesNum(movimiento).val_int_clase

        Y[Y == 0] = tipo_movimiento

        X[inc_rows_start:inc_rows_end, 0] = Y[:, 0]
        X[inc_rows_start:inc_rows_end, 1:len(vector.transpose()) + 1] = vector

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

    vector = pd.read_csv(pca_file)

    df_X = pd.DataFrame(vector)
    df_X.to_csv(r'' + 'datos_nuevos' + '/' + file_name, index=False, header=False)
