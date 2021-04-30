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
Para la ejecución de cada uno de los pasos
1.	read_file_py.py. Lectura del archivo binario .dat y su conversión a .csv.
  *	csi_headers.csv. Encabezados para cada una de las columnas de los datos.
  *	Amplitude.py. Función integrada en  read_file_py.py.  para la conversión de los datos obtenidos de CSI, a datos de Amplitud.

Almacena la matriz generada en la carpeta “/datos_crudos”.

2.	preprocesamiento.py. Imputación de Datos, Eliminación de Ruido, y Normalización.
  *	DataBaseConnection.py. Clase para la conexión con la base de datos en MySQL Server. Es instanciada por preprocesamiento.py.
  *	ClasesNumericas.py. Clase para clasificar los datos de entrenamiento, que de antemano ya se sabe a qué movimiento pertenecen. Es instanciada por preprocesamiento.py.

Toma como entrada la matriz de “/datos_crudos”. Almacena la matriz generada en la carpeta “/preprocesados”; de manera opcional, realiza la inserción en la base de datos de los valores máximos y mínimos de cada columna de la matriz de entrada, para futuros análisis.

pca.py. Extracción de los Principales Componentes (PCA). 
Toma como entrada la matriz de “/preprocesados”. Almacena la matriz generada en la carpeta “/pca” y los heatmaps y gráficos generados de Amplitud y 6 PCA, en la carpeta “/pca/images”.

3.	generarMatrizX.py. Extracción del Dominio del Tiempo de cada matriz de PCA, de cada movimiento, y construcción de una nueva matriz, que será dividida en training y testing.

Toma como entrada la matriz de “/pca”. Almacena la matriz de training X y el vector de clases Y generados, en la carpeta “/trn_tst”.

4.	EnsambleClasificadores.py. Script para comprobar la tasa de reconocimiento a partir de cada uno de los modelos de clasificación: Nearest Neighbors, Gaussian Naive Bayes, Linear Discriminant Analysis, Quadratic Discriminant Analysis, Support Vector Machine y Neural Network.

Toma como entrada la matriz de training X y el Vector de Clases Y, separa los datos en training y testing, con el fin de predecir el tipo de movimiento de los datos de testing y compararlo contra su valor real. Como salida se obtiene impreso en pantalla el porcentaje de reconocimiento de los datos de test.

5.	Classification.py. Script para realizar la clasificación de datos nuevos, que nunca ha visto el modelo, empleando algoritmos de Nearest Neighbors, Support Vector Machine y Neural Network.

Toma como entrada el vector generado con “/scripts_de_apoyo/AtribDomTiempo.py”, genera como salida el movimiento clasificado, impreso en pantalla. Utiliza la matriz de training X y el Vector de Clases Y, porque a partir de ellos se crean los respectivos modelos.


Para la ejecución de los pasos de manera integrada
1.	AssembledModel.py. Script del proceso completo, desde la lectura del archivo .dat, hasta la clasificación, a partir de datos nunca antes vistos, tomando como base la matriz de datos training “/trn_tst/X.csv”, y el vector de datos “/trn_tst/Y.csv”, empleando algoritmos de Nearest Neighbors, Support Vector Machine y Neural Network. 
  *	custom_features.json. Script que permite obtener los atributos en el dominio del tiempo.  Es llamado por AssembledModel.py

Toma como entrada el archivo un archivo nuevo .dat, y genera como salida, la clasificación en pantalla del tipo de movimiento introducido al algoritmo, de manera opcional, genera los heatmaps y gráficos de Amplitudes y PCA si se le indica que los guarde, al inicio de la ejecución del algoritmo. Utiliza la matriz de training X y el Vector de Clases Y, porque a partir de ellos se crean los respectivos modelos.
