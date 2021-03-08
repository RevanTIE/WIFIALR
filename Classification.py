from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm
import statistics as stat

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from DataBaseConnection import DataBase

'''
Script para realizar la clasificación en datos nuevos, que nunca ha visto el modelo
'''

contador = [1, 2, 3, 4, 5, 6]
X_train = pd.read_csv("trn_tst/X.csv", names=contador)
Y_vector = pd.read_csv("trn_tst/Y.csv", names=[0])

y_train = np.ravel(Y_vector)

root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
X_new_list = askopenfilename(title='Choose a file', initialdir='datos_nuevos', filetypes = (("CSV Files","*.csv"),))
X_test = pd.read_csv(X_new_list)

database = DataBase()

"""
Nearest Neighbors Classification
"""
n_neighbors = 7
neigh = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

y_pred_neigh = neigh.fit(X_train, y_train).predict(X_test)
KNN_moda = stat.mode(y_pred_neigh)
KNN_mov = database.select_movimientos(KNN_moda)

print('Nearest Neighbors : %d' % (KNN_mov))
"""
Support Vector Machine
"""
supVM = svm.SVC(kernel='linear', C=1)
y_pred_supVM = supVM.fit(X_train, y_train).predict(X_test)
SVM_moda = stat.mode(y_pred_supVM)
SVM_mov = database.select_movimientos(SVM_moda)

print('Support Vector Machine : %d' % (SVM_mov))
"""
Neural Network
"""
neuNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
y_pred_neuNet = neuNet.fit(X_train, y_train).predict(X_test)
NeuralN_moda = stat.mode(y_pred_neuNet)
NeuralN_mov = database.select_movimientos(NeuralN_moda)

print('Neural Network : %d' % (NeuralN_mov))

database.close()

vector_modas = [KNN_moda, SVM_moda, NeuralN_moda]
mov_predecido =  stat.mode(vector_modas)


try:
   database = DataBase()
   database.select_alertas(mov_predecido)
   database.close()
            
except Exception as e:
    raise




