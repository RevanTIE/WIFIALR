import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tkinter import re


def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def visualize(path1, file_nam):
    # data import
    data = pd.read_csv(path1).values
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

    # xmin = 0
    # xmax = 20000
    # plt


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
    """
    plt.figure(figsize=(18, 30))
    # Spectrogram(STFT)
    plt.subplot(611)
    Pxx, freqs, bins, im = plt.specgram(pca_data2[:, 0], NFFT=128, Fs=1000, noverlap=1, cmap="jet", vmin=-100, vmax=20)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram(STFT)")
    plt.colorbar(im)
    plt.xlim(0, 10)
    plt.ylim(0, 100)

    plt.subplot(612)
    Pxx, freqs, bins, im = plt.specgram(pca_data2[:, 1], NFFT=128, Fs=1000, noverlap=1, cmap="jet", vmin=-100, vmax=20)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram(STFT)")
    plt.colorbar(im)
    plt.xlim(0, 10)
    plt.ylim(0, 100)

    plt.subplot(613)
    Pxx, freqs, bins, im = plt.specgram(pca_data2[:, 2], NFFT=128, Fs=1000, noverlap=1, cmap="jet", vmin=-100, vmax=20)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram(STFT)")
    plt.colorbar(im)
    plt.xlim(0, 10)
    plt.ylim(0, 100)

    plt.subplot(614)
    Pxx, freqs, bins, im = plt.specgram(pca_data2[:, 3], NFFT=128, Fs=1000, noverlap=1, cmap="jet", vmin=-100, vmax=20)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram(STFT)")
    plt.colorbar(im)
    plt.xlim(0, 10)
    plt.ylim(0, 100)

    plt.subplot(615)
    Pxx, freqs, bins, im = plt.specgram(pca_data2[:, 4], NFFT=128, Fs=1000, noverlap=1, cmap="jet", vmin=-100, vmax=20)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram(STFT)")
    plt.colorbar(im)
    plt.xlim(0, 10)
    plt.ylim(0, 100)

    plt.subplot(616)
    Pxx, freqs, bins, im = plt.specgram(pca_data2[:, 5], NFFT=128, Fs=1000, noverlap=1, cmap="jet", vmin=-100, vmax=20)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram(STFT)")
    plt.colorbar(im)
    plt.xlim(0, 10)
    plt.ylim(0, 100)

    plt.show()
    """

    return pca_data2

#Se añaden los encabezados
csv_headers = "csi_headers.csv"
csv_cols = pd.read_csv(csv_headers)[1:91]
csv_col_list = csv_cols["Column_Names"].tolist()

root = Tk() #Elimina la ventana de Tkinter
root.withdraw() #Ahora se cierra
file_path = askopenfilenames(parent=root, title='Choose a file', initialdir='preprocesados', filetypes = (("CSV Files","*.csv"),)) #Se abre el explorador de archivos y se guarda la selección

for i in range(len(file_path)):
    splitted = file_path[i].split("/")
    file_name = splitted[-1]
    folder_name = file_path[i].replace(file_name,'')

    short_name = file_name.split(".csv")[-2]

    #Nombre del archivo
    pca_data2= visualize(file_path[i], short_name)

    pcaDataFrame = pd.DataFrame(pca_data2, columns=csv_col_list)
    pcaDataFrame.to_csv(r''+ 'pca' +'\pca_' + file_name, index = False, header=True)