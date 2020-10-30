# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 00:03:20 2020

@author: elohe

Implementar el 2do método de construcción de X, Y

1. Se mandan a llamar X y Y.
2. Se revuelven los datos y sus respectivos indices.
3. Selección de modelos y sus parámetros.
4. Se aplica 10 fold cross validation sobre cada uno de los modelos para saber 
        la mejor tasa de reconocimiento por fold.

"""
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm

#df_X = pd.read_csv("trn_tst/X.csv")
#df_Y = pd.read_csv("trn_tst/Y.csv")

"""
with open('trn_tst/X.csv', newline='') as f:
    csv_X = csv.reader(f)
    list_X= list(csv_X)

for i in range(len(list_X)):
    mapeo = map(float, list_X[i])
    lista_X= list(mapeo)
    
    for j in range(len(lista_X)):
        X_t.append(lista_X[j])
"""
 
with open('trn_tst/Y.csv', newline='') as g:
    csv_Y = csv.reader(g)
    list_Y= list(csv_Y)

map_object_Y  = map(float, list_Y[0])
Y = [int(i) for i in(list(map_object_Y))]

contador = [] 

for i in range(len(Y)):
    contador.append(i)

df_X = pd.read_csv("trn_tst/X.csv", names=contador)

X = df_X.transpose()

X_iris, Y_iris = datasets.load_iris(return_X_y=True)
##X.shape, Y.shape

# Se dividen en training y test, mientras que se revuelven los datos.
"""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
clf = svm.SVC(kernel='poly', C=1, gamma='scale').fit(X_train, y_train)
scores = clf.score(X_test, y_test)

"""
# Selección del modelo y parámetros: Support Vector Machine
# Sustituir por un shuffle diferente, que no divida en training/test, y hacer pruebas de ambos
cv_suffled = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0) 
clf = svm.SVC(kernel='linear', C=1)

#Aplicación de cross fold validation
scores = cross_val_score(clf, X, Y, cv=5)


