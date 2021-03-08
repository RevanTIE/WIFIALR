"""
Script para la clasificación de los datos nuevos que lleguen
Nota: Falta incluir funciones de la conversión de datos de .dat a .csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from DataBaseConnection import DataBase
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import svm
import statistics as stat


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

def preprocesamiento(file_p, csv_headers):
    # Se añaden los encabezados
    csv_cols = pd.read_csv(csv_headers)
    csv_col_list = csv_cols["Column_Names"].tolist()
    trn = pd.read_csv(file_p, names=csv_col_list)
    trn_tim = trn['timestamp']

    # Se convierte en dataframe
    trn_tim_df = pd.DataFrame(trn_tim)
    # Se calculan los segundos de actividad
    tim_normalizado = normalizar(trn_tim_df) * (trn_tim.max() - trn_tim.min())

    # Imputación de datos
    # Se pone nan como 0
    trn_NaN_2_0 = trn.fillna(1.0000e-5)

    # Eliminación de ruido
    trn_matrix = trn_NaN_2_0.values
    #rows_matrix = len(trn_matrix)
    cols_matrix = len(np.transpose(trn_matrix))

    trn_sin_ruido = trn_matrix[:, 0:cols_matrix]  # Sin el timestamp, 180 variables
    trn_sin_ruido_collected = trn_sin_ruido * 0

    for dat in range(cols_matrix):
        trn_sin_ruido_collected[:, dat] = ruido(trn_sin_ruido[:, dat])

    sin_ruido_df = pd.DataFrame(trn_sin_ruido_collected, columns=csv_col_list)

    trn_normalizado = normalizar(sin_ruido_df)
    trn_normalizado['timestamp'] = tim_normalizado['timestamp']

    return trn_normalizado


def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def fpca(datos, file_nam):
    # data import
    data = datos.values
    amp = data[1:len(data), 1:91]
    # plt
    
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
    
    pcaDataFrame = pd.DataFrame(pca_data2, columns=csv_col_list)
    
    return pcaDataFrame


def fclasificacion(pca_matrix):
    contador = [1, 2, 3, 4, 5, 6]
    X_train = pd.read_csv("trn_tst/X.csv", names=contador)
    Y_vector = pd.read_csv("trn_tst/Y.csv", names=[0])

    y_train = np.ravel(Y_vector)
    X_test = pca_matrix
    
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
file_path = askopenfilenames(parent=root, title='Choose a file', initialdir='datos_crudos',
                               filetypes=(("CSV Files", "*.csv"),))

# Se añaden los encabezados
csv_headers = "csi_headers.csv"
csv_cols = pd.read_csv(csv_headers)[1:91]
csv_col_list = csv_cols["Column_Names"].tolist()

for i in range(len(file_path)):
    splitted = file_path[i].split("/")
    file_name = splitted[-1]

    short_name = file_name.split(".csv")[-2]

    print('Pruebas de : %s' % short_name)

    datos_preprocesados = preprocesamiento(file_path[i], csv_headers)
    datos_pca = fpca(datos_preprocesados, short_name)

    pca_matrix = datos_pca.iloc[:, 0:6]

    mov_predecido = fclasificacion(pca_matrix)

    try:
        database = DataBase()
        database.select_alertas(mov_predecido)
        database.close()

    except Exception as e:
        raise


