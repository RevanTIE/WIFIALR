"""
Script del proceso completo implementando el modelo construido
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilenames
from DataBaseConnection import DataBase
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm
import statistics as stat

from scipy.signal import savgol_filter
from sklearn.impute import KNNImputer

import tsfel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ClasesNumericas import ClasesNum
from sklearn.naive_bayes import GaussianNB

from CSIKit.reader import get_reader
import Amplitude
from numpy import inf

def extracting_csi(file_path):
    splitted = file_path.split("/")
    file_name = splitted[-1]

    my_reader = get_reader(file_path)
    csi_data = my_reader.read_file(file_path, scaled=True)
    (csi_1, csi_2, csi_3) = Amplitude.get_CSI_Frames(csi_data)

    csi_matrix_inversa_1 = csi_1.transpose()
    csi_matrix_inversa_2 = csi_2.transpose()
    csi_matrix_inversa_3 = csi_3.transpose()

    timestamp_vector = csi_data.timestamps

    csi_amp_matrix = np.zeros([csi_data.expected_frames, 90])
    csi_amp_matrix[:, 0:30] = csi_matrix_inversa_1
    csi_amp_matrix[:, 30:60] = csi_matrix_inversa_2
    csi_amp_matrix[:, 60:90] = csi_matrix_inversa_3

    csi_amp_matrix[csi_amp_matrix == -inf] = np.nan

    csvNewFile = np.zeros([csi_data.expected_frames, len(np.transpose(csi_amp_matrix)) + 1])
    time_v = np.ravel(timestamp_vector)

    csvNewFile = np.c_[time_v, csi_amp_matrix]
    dFCsv = pd.DataFrame(csvNewFile)
    return dFCsv

"""
    Atributos en el dominio del tiempo
"""
def AtribDomTiempo(df):
    # Retrieves a pre-defined feature configuration file to extract all available features
    cfg = tsfel.get_features_by_domain("statistical", "custom_features.json")

    # Extract features
    extracted_features = tsfel.time_series_features_extractor(cfg, df)
    return extracted_features

##Función de normalización manual
def normalizar(df):
    result = df.copy()
    for dato in df.columns:
        max_value = df[dato].max()
        min_value = df[dato].min()

        result[dato] = (df[dato] - min_value) / (max_value - min_value)
    return result


##Función de eliminación de ruido (Por vector)
def ruido(x):
    ##   Savitzky-Golay params
    # len(x)
    window_length  = 249
    polyorder = 5
    
    filtro = savgol_filter(x, window_length, polyorder, mode='nearest')

    return filtro

def preprocesamiento(file, csv_col_list):
    # Se añaden los encabezados
    trn = file
    trn.columns = csv_col_list
    trn_tim = trn['timestamp']

    # Se convierte en dataframe
    trn_tim_df = pd.DataFrame(trn_tim)

    # Imputación básica en base a estadísticos - Imputación por Knn
    knn = KNNImputer(n_neighbors=5)
    trn_matrix = knn.fit_transform(trn)
    
    ## simple = SimpleImputer().fit(trn)
    ## trn_NaN_2_0 = simple.transform(trn) ## trn.fillna(1.0000e-5)

    # Eliminación de ruido
    # trn_matrix = trn_NaN_2_0
    rows_matrix = len(trn_matrix)
    cols_matrix = len(np.transpose(trn_matrix))

    trn_sin_ruido = trn_matrix[:, 0:cols_matrix]  # Sin el timestamp, 90 variables
    trn_sin_ruido_collected = trn_sin_ruido * 0

    for dat in range(cols_matrix):
        trn_sin_ruido_collected[:, dat] = ruido(trn_sin_ruido[:, dat])

    sin_ruido_df = pd.DataFrame(trn_sin_ruido_collected, columns=csv_col_list)

    trn_normalizado = sin_ruido_df ## normalizar(sin_ruido_df)
    trn_normalizado['timestamp'] = trn_tim_df

    return trn_normalizado


def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def images(datos, file_nam):

    # data import
    data = datos.values
    amp = data[1:len(data), 1:91]
    # plt

    if guardarImagenes == "S":
        ## Amplitud de las tres antenas
        fig = plt.figure(figsize=(18, 15))
        ax1 = plt.subplot(311)
        plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
        plt.xlabel("Packet index")
        plt.ylabel("Subcarrier index")
        ax1.set_title("Antenna A")
        plt.colorbar()

        ax2 = plt.subplot(312)
        plt.imshow(amp[:, 30:59].T, interpolation="nearest", aspect="auto", cmap="jet")
        plt.xlabel("Packet index")
        plt.ylabel("Subcarrier index")
        ax2.set_title("Antenna B")
        plt.colorbar()

        ax3 = plt.subplot(313)
        plt.imshow(amp[:, 60:89].T, interpolation="nearest", aspect="auto", cmap="jet")
        plt.xlabel("Packet index")
        plt.ylabel("Subcarrier index")
        ax3.set_title("Antenna C")
        plt.colorbar()
        plt.savefig('images/' + file_nam + '_amplitude.png')

        ## 2 subcarriers de cada antena

        fig3 = plt.figure(figsize=(18, 30))

        ax1 = plt.subplot(611)
        plt.plot(amp[:, 0])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,0])
        ax1.set_title("Antenna A 1st subcarrier")

        ax2 = plt.subplot(612)
        plt.plot(amp[:, 1])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,1])
        ax2.set_title("Antenna A 2nd subcarrier")

        ax3 = plt.subplot(613)
        plt.plot(amp[:, 30])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,2])
        ax3.set_title("Antenna B 1st subcarrier")

        ax4 = plt.subplot(614)
        plt.plot(amp[:, 31])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,3])
        ax4.set_title("Antenna B 2nd subcarrier")

        ax5 = plt.subplot(615)
        plt.plot(amp[:, 60])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,4])
        ax5.set_title("Antenna C 1st subcarrier")

        ax6 = plt.subplot(616)
        plt.plot(amp[:, 61])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,5])
        ax6.set_title("Antenna C 2nd subcarrier")

        plt.savefig('images/'+ file_nam +'_subcarriers.png')


def fclasificacion(pca_vector):
    contador = []
    for i in range(len(pca_vector.transpose())):
        contador.append(i + 1)

    X_train = pd.read_csv("trn_tst/X.csv", names=contador)
    Y_vector = pd.read_csv("trn_tst/Y.csv", names=[0])

    y_train = np.ravel(Y_vector)
    X_test = pca_vector
    
    database = DataBase()

    """
    Nearest Neighbors Classification
    """
    n_neighbors = 7
    neigh = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    y_pred_neigh = neigh.fit(X_train, y_train).predict(X_test)
    KNN_mov = database.select_movimientos(y_pred_neigh[0])

    print('Nearest Neighbors : %s' % KNN_mov)
    """
    Support Vector Machine
    """
    supVM = svm.SVC(kernel='linear', C=1)
    y_pred_supVM = supVM.fit(X_train, y_train).predict(X_test)
    SVM_mov = database.select_movimientos(y_pred_supVM[0])

    print('Support Vector Machine : %s' % SVM_mov)

    """
    Gaussian Naive Bayes
    """
    gnb = GaussianNB()
    y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
    Gaussian_mov = database.select_movimientos(y_pred_gnb[0])

    print('Gaussian Naive Bayes: %s' % Gaussian_mov)


    """
    Neural Network
    """
    neuNet = MLPClassifier(solver='sgd', alpha=0.0001, random_state=1)
    y_pred_neuNet = neuNet.fit(X_train, y_train).predict(X_test)
    NeuralN_mov = database.select_movimientos(y_pred_neuNet[0])

    print('Neural Network : %s' % NeuralN_mov)

    """
    Linear Discriminant Analysis
    """
    lda = LinearDiscriminantAnalysis()
    y_pred_lda = lda.fit(X_train, y_train).predict(X_test)
    lda_mov = database.select_movimientos(y_pred_lda[0])

    print('Linear Discriminant Analysis: %s' % lda_mov)
    
    database.close()

    try:
        vector_modas = [y_pred_neigh[0], y_pred_supVM[0], y_pred_neuNet[0], y_pred_lda[0], y_pred_gnb[0]]
        mov_predecido = stat.mode(vector_modas)

    except Exception as e:
        mov_predecido = y_pred_supVM[0]



    return mov_predecido


#Se abre una ventana de dialogo para solicitar el archivo csv
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra

"""
    Preguntar si se desea tratar la lista como elementos de Testing para generar matriz X y clases Y de pruebas
"""
elementosTraining = input("¿Desea Tratar los datos como testing? S = SI, N = NO: ")
"""
    Preguntar si se desea almacenar las imagenes de las señales
"""
guardarImagenes = input("¿Desea almacenar las imagenes de las señales? S = SI, N = NO: ")

file_path = askopenfilenames(parent=root, title='Choose a file', initialdir='datos_crudos',
                               filetypes=(("DAT Files", "*.dat"),))

# Se añaden los encabezados
csv_headers = "csi_headers.csv"
csv_cols = pd.read_csv(csv_headers)[0:91]
csv_col_list = csv_cols["Column_Names"].tolist()

Y_testing = []
file_len = len(file_path)
nam = tuple(list(range(360)))
X_testing = np.zeros([file_len, 360])


for i in range(file_len):
    splitted = file_path[i].split("/")
    file_name = splitted[-1]

    short_name = file_name.split(".dat")[-2]

    print('Pruebas de : %s' % short_name)

    datos_crudos = extracting_csi(file_path[i])
    datos_preprocesados = preprocesamiento(datos_crudos, csv_col_list)
    images(datos_preprocesados, short_name)
    datos_sin_timestamp =  datos_preprocesados.drop(['timestamp'], axis=1)  ## Descomentar
    
    vector = AtribDomTiempo(datos_sin_timestamp.iloc[:]) ## Descomentar
    
    if elementosTraining == "S":
        movimiento = short_name.split("_")[-4]
        tipo_movimiento = ClasesNum(movimiento).val_int_clase
        
        Y_testing.append(tipo_movimiento)
        X_testing[i, :] = vector.values
    
    else:
        mov_predecido = fclasificacion(vector)

        try:
            database = DataBase()
            movimiento = database.select_alertas(mov_predecido)
            print (movimiento)
            #messagebox.showwarning("MOVIMIENTO DETECTADO", movimiento)  ## Descomentar para pop up
            database.close()

        except Exception as e:
            raise
            

if elementosTraining == "S":
    X_testing_df = pd.DataFrame(X_testing)
    Y_testing_df = pd.DataFrame(Y_testing)

    X_testing_df.to_csv(r'' + 'datos_nuevos' + '/X_test.csv', index=False, header=False)
    Y_testing_df.to_csv(r'' + 'datos_nuevos' + '/Y_test.csv', index=False, header=False)