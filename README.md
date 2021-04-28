# CSIProject

El siguiente proyecto incorpora scripts para el manejo de datasets, resultado de obtención de señales CSI desde Tx hasta Rx.
* datos_crudos. Carpeta que almacena matrices, de los datos convertidos de .dat a .csv.
* img. Imagenes relacionadas al proyecto.
* librerias. Librerías de GitHub de terceros para el apoyo del proyecto.
* pca. Matrices como resultado de aplicar PCA a los datos preprocesados.
* preprocesados. Matrices como resultado de preprocesar los datos crudos.
* trn_tst. Matriz de los datos de training X, y el vector de clases Y.
* csi_headers.csv. Encabezados de las matrices.

Orden de ejecución por pasos para Generación de Matriz de training X y Vector de clases Y
1.	read_file_py.py. Lectura del archivo binario .dat y su conversión a .csv.
2.	Amplitude.py. Función integrada en  read_file_py.py.  para la conversión de los datos obtenidos de CSI, a datos de Amplitud.
3.	preprocesamiento.py. Imputación de Datos, Eliminación de Ruido, y Normalización.
4.	pca.py. Extracción de los Principales Componentes (PCA).
5.	generarMatrizX.py. Extracción del Dominio del Tiempo de cada matriz de PCA, de cada movimiento, y construcción de una nueva matriz, que será dividida en training y testing.
