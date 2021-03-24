"""Atributos en el dominio del tiempo
Extrae los atributos por cada una de las columnas de PCA:
- Mean
- Median
- Root Mean Square Error
- Variance

Resultado: Un vector de fila, de 1 x 360 por movimiento
"""

import tsfel
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

root = Tk()  # Elimina la ventana de Tkinter
root.withdraw()  # Ahora se cierra

# load dataset
pca_file = askopenfilename(parent=root, title='Choose a file', initialdir='pca',
                               filetypes=(("CSV Files", "*.csv"),))
splitted = pca_file.split("/")
file_name = splitted[-1]

df = pd.read_csv(pca_file)

# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain("statistical", "custom_features.json")

# Extract features
X = tsfel.time_series_features_extractor(cfg, df)