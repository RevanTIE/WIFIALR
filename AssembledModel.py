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
import tsfel

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
def ruido(i):
    mirror_i = np.transpose(i)
    w = 3
    ven = w

    prev = mirror_i[0:ven - 1]
    prev_mean = np.mean(prev)

    modified_length = len(mirror_i) - 2
    z_table = []

    ##Para length del primer valor hasta el final del vector
    for data in range(modified_length):
        y = mirror_i
        if ven <= len(mirror_i):
            z = y[data:ven]
            ven = ven + 1

        acumulado = np.mean(z)
        z_table.append(acumulado)

    penultimo = mirror_i[-2]
    ultimo = mirror_i[-1]
    last = [penultimo, ultimo]
    last_mean = np.mean(last)

    z_table.insert(0, prev_mean)
    z_table.append(last_mean)

    return z_table

def preprocesamiento(file, csv_col_list):
    # Se añaden los encabezados
    trn = file
    trn.columns = csv_col_list
    trn_tim = trn['timestamp']

    # Se convierte en dataframe
    trn_tim_df = pd.DataFrame(trn_tim)

    # Imputación de datos
    inf_estadistica_trn = trn.describe()  # Por lo tanto existen datos faltantes.

    # Razones por las que faltan datos:
    # 1. Se puede optimizar el procesa de extracción de los datos
    # 2. Debido a la recolección de los datos

    # Se pone nan como 0
    trn_NaN_2_0 = trn.fillna(1.0000e-5)  # Se puede mejorar la imputación de datos empleando otra función.

    # Eliminación de ruido
    trn_matrix = trn_NaN_2_0.values
    rows_matrix = len(trn_matrix)
    cols_matrix = len(np.transpose(trn_matrix))

    trn_sin_ruido = trn_matrix[:, 0:cols_matrix]  # Sin el timestamp, 90 variables
    trn_sin_ruido_collected = trn_sin_ruido * 0

    for dat in range(cols_matrix):
        trn_sin_ruido_collected[:, dat] = ruido(trn_sin_ruido[:, dat])

    sin_ruido_df = pd.DataFrame(trn_sin_ruido_collected, columns=csv_col_list)

    trn_normalizado = normalizar(sin_ruido_df)
    trn_normalizado['timestamp'] = trn_tim_df

    return trn_normalizado


def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def fpca(datos, file_nam):
    # data import
    data = datos.values
    amp = data[1:len(data), 1:91]
    # plt

    if guardarImagenes == "S":

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
        plt.savefig('pca/images/' + file_nam + '_amplitude.png')
    
    # Initializing variables
    constant_offset = np.empty_like(amp)
    filtered_data = np.empty_like(amp)

    # Calculating the constant offset (moving average 20 seconds)
    for i in range(1, len(amp[0])):
        constant_offset[:, i] = moving_average(amp[:, i], len(amp[0]))

    # Calculating the filtered data (substract the constant offset)
    filtered_data = amp - constant_offset

    # Smoothing (moving average 0.01 seconds)
    for i in range(1, len(amp[0])):
        filtered_data[:, i] = moving_average(filtered_data[:, i], 10)
    # Calculate correlation matrix (90 * 90 dim)
    cov_mat2 = np.cov(filtered_data.T)
    # Calculate eig_val & eig_vec
    eig_val2, eig_vec2 = np.linalg.eig(cov_mat2)
    # Sort the eig_val & eig_vec
    idx = eig_val2.argsort()[::-1]
    eig_val2 = eig_val2[idx]
    eig_vec2 = eig_vec2[:, idx]
    # Calculate H * eig_vec
    pca_data2 = filtered_data.dot(eig_vec2)

    if guardarImagenes == "S":
        fig3 = plt.figure(figsize=(18, 30))

        ax1 = plt.subplot(611)
        plt.plot(pca_data2[:, 0])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,0])
        ax1.set_title("PCA 1st component")

        ax2 = plt.subplot(612)
        plt.plot(pca_data2[:, 1])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,1])
        ax2.set_title("PCA 2nd component")

        ax3 = plt.subplot(613)
        plt.plot(pca_data2[:, 2])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,2])
        ax3.set_title("PCA 3rd component")

        ax4 = plt.subplot(614)
        plt.plot(pca_data2[:, 3])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,3])
        ax4.set_title("PCA 4th component")

        ax5 = plt.subplot(615)
        plt.plot(pca_data2[:, 4])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,4])
        ax5.set_title("PCA 5th component")

        ax6 = plt.subplot(616)
        plt.plot(pca_data2[:, 5])
        plt.xlabel("Time[s]")
        plt.ylabel("Observation values")
        # plt.plot(pca_data2[2500:17500,5])
        ax6.set_title("PCA 6th component")

        plt.savefig('pca/images/'+ file_nam +'_PCA.png')
    
    pcaDataFrame = pd.DataFrame(pca_data2, columns=csv_col_list[1:91])

    if elementosTraining == "S":        #Posteriormente se tiene que utilizar “/scripts_de_apoyo/AtribDomTiempo.py”
        pcaDataFrame.to_csv(r'' + 'pca' + '/pca_' + file_nam + '.csv', index=False, header=True)
    
    return pcaDataFrame


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
    KNN_moda = stat.mode(y_pred_neigh)
    KNN_mov = database.select_movimientos(KNN_moda)

    print('Nearest Neighbors : %s' % KNN_mov)
    """
    Support Vector Machine
    """
    supVM = svm.SVC(kernel='linear', C=1)
    y_pred_supVM = supVM.fit(X_train, y_train).predict(X_test)
    SVM_moda = stat.mode(y_pred_supVM)
    SVM_mov = database.select_movimientos(SVM_moda)

    print('Support Vector Machine : %s' % SVM_mov)
    """
    Neural Network
    """
    neuNet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    y_pred_neuNet = neuNet.fit(X_train, y_train).predict(X_test)
    NeuralN_moda = stat.mode(y_pred_neuNet)
    NeuralN_mov = database.select_movimientos(NeuralN_moda)

    print('Neural Network : %s' % NeuralN_mov)
    
    database.close()

    vector_modas = [KNN_moda, SVM_moda, NeuralN_moda]
    mov_predecido = stat.mode(vector_modas)

    return mov_predecido


#Se abre una ventana de dialogo para solicitar el archivo csv
root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra

"""
    Preguntar si se desea tratar la lista como elementos de Trainig para generar matrices de PCA
"""
elementosTraining = input("¿Desea Tratar los datos como training? S = SI, N = NO: ")
"""
    Preguntar si se desea almacenar las imagenes en la carpeta en PCA
"""
guardarImagenes = input("¿Desea almacenar las imagenes de PCA? S = SI, N = NO: ")

file_path = askopenfilenames(parent=root, title='Choose a file', initialdir='datos_crudos',
                               filetypes=(("DAT Files", "*.dat"),))

# Se añaden los encabezados
csv_headers = "csi_headers.csv"
csv_cols = pd.read_csv(csv_headers)[0:91]
csv_col_list = csv_cols["Column_Names"].tolist()

for i in range(len(file_path)):
    splitted = file_path[i].split("/")
    file_name = splitted[-1]

    short_name = file_name.split(".dat")[-2]

    print('Pruebas de : %s' % short_name)

    datos_crudos = extracting_csi(file_path[i])
    datos_preprocesados = preprocesamiento(datos_crudos, csv_col_list)
    datos_pca = fpca(datos_preprocesados, short_name)

    if elementosTraining == "N":
        vector = AtribDomTiempo(datos_pca.iloc[:])
        mov_predecido = fclasificacion(vector)

        try:
            database = DataBase()
            movimiento = database.select_alertas(mov_predecido)
            print (movimiento)
            messagebox.showwarning("MOVIMIENTO DETECTADO", movimiento)
            database.close()

        except Exception as e:
            raise

