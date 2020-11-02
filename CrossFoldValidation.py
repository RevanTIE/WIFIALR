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

contador = [1, 2, 3, 4, 5, 6]

X = pd.read_csv("trn_tst/X.csv", names=contador)
Y = pd.read_csv("trn_tst/Y.csv", names=[0])

"""
X_iris, Y_iris = datasets.load_iris(return_X_y=True)
"""
##X.shape, Y.shape

# Se dividen en training y test, mientras que se revuelven los datos.
"""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
clf = svm.SVC(kernel='poly', C=1, gamma='scale').fit(X_train, y_train)
scores = clf.score(X_test, y_test)

"""
# Selección del modelo y parámetros: Support Vector Machine
# Sustituir por un shuffle diferente, que no divida en training/test, y hacer pruebas de ambos
cv_suffled = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) 
clf = svm.SVC(kernel='linear', C=1)

#Aplicación de cross fold validation
scores = cross_val_score(clf, X, np.ravel(Y), cv=cv_suffled)


