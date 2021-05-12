from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm

'''
Script para probar la tasa de reconocimiento de cada uno de los modelos de clasificación
'''

contador = [1, 2, 3, 4, 5, 6]
X = pd.read_csv("trn_tst/X.csv", names=contador)
Y_vector = pd.read_csv("trn_tst/Y.csv", names=[0])

y = np.ravel(Y_vector)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("TEST ACCURACY")

"""
Nearest Neighbors Classification
"""
n_neighbors = 7
neigh = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

y_pred_neigh = neigh.fit(X_train, y_train).predict(X_test)
accuracy_neigh = ((y_test == y_pred_neigh).sum() / X_test.shape[0]) * 100
print('Nearest Neighbors : %.2f de un total de %d' % (accuracy_neigh, X_test.shape[0]))

"""
Gaussian Naive Bayes
"""
gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
accuracy_gnb = ((y_test == y_pred_gnb).sum() / X_test.shape[0]) * 100
print('Gaussian Naive Bayes: %.2f de un total de %d' % (accuracy_gnb, X_test.shape[0]))

"""
Linear Discriminant Analysis
"""
lda = LinearDiscriminantAnalysis()
y_pred_lda = lda.fit(X_train, y_train).predict(X_test)
accuracy_lda = ((y_test == y_pred_lda).sum() / X_test.shape[0]) * 100
print('Linear Discriminant Analysis: %.2f de un total de %d' % (accuracy_lda, X_test.shape[0]))

"""
Quadratic Discriminant Analysis
"""
qda = QuadraticDiscriminantAnalysis()
y_pred_qda = qda.fit(X_train, y_train).predict(X_test)
accuracy_qda = ((y_test == y_pred_qda).sum() / X_test.shape[0]) * 100
print('Quadratic Discriminant Analysis: %.2f de un total de %d' % (accuracy_qda, X_test.shape[0]))

"""
Neural Network
"""
neuNet = clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
y_pred_neuNet = neuNet.fit(X_train, y_train).predict(X_test)
accuracy_neuNet= ((y_test == y_pred_neuNet).sum() / X_test.shape[0]) * 100
print('Neural Network: %.2f de un total de %d' % (accuracy_neuNet, X_test.shape[0]))

"""
Support Vector Machine
"""
supVM = svm.SVC(kernel='linear', C=1)
y_pred_supVM = supVM.fit(X_train, y_train).predict(X_test)
accuracy_supVM = ((y_test == y_pred_supVM).sum() / X_test.shape[0]) * 100
print('Support Vector Machine: %.2f de un total de %d' % (accuracy_supVM, X_test.shape[0]))