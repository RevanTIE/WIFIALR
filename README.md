# WIFIALR

El siguiente proyecto incorpora scripts para el manejo de datasets, resultado de obtención de señales CSI desde Tx hasta Rx.
* datos_crudos. Carpeta que almacena matrices multidimensionales de datos sin procesar en formato .dat.
* datos_nuevos. Almacena las matrices de "X_test.csv" y "Y_test.csv" generadas por el script "AssembledModel.py" al seleccionar que SI se desea tratar los datos como testing.
* db_schemas. Almacena el Modelo Entidad-Relación de la base de datos, así como scripts de generación de la base de datos desde cero y sus respectivas actualizaciones. NOTA: El script "crearModelo_version_02.sql" genera la versión actual de la base de datos con éxito, solo basta ejecutar este script.
* images. Contiene gráficos Heatmaps de la Amplitud de la señal presente en cada una de las antenas del dispositivo Rx, así como gráficos que representan señales de seis subcarriers por movimiento realizado.
* librerias. Librerías de GitHub de terceros para el apoyo del proyecto.
* pca (Ya no es utilizada). Matrices como resultado de aplicar PCA a los datos preprocesados. 
* preprocesados (Ya no es utilizada). Matrices como resultado de preprocesar los datos crudos.
* trn_tst. Matriz de los datos de training X, y el vector de clases Y.
* scripts_de_apoyo. Algunos scripts que sirvieron durante el desarrollo de la Aplicación para realizar pruebas de variables en el dominio del tiempo, Cross-Fold Validation y gráficos PCA.
* csi_headers.csv. Encabezados de las matrices.

Orden de ejecución por pasos para Generación de Matriz de training X y Vector de clases Y. 
NOTA: Esta es la manera rudimentaria de generación de alertas que se utilizó durante el desarrollo del proyecto, con la finalidad de observar los resultados de las matrices generadas en cada una de las etapas por las que pasan los datos. Actualmente ya no se utiliza más que para fines de observación.

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


Para la ejecución de los pasos de manera integrada, es decir, el actual y correcto funcionamiento del software, se lleva a cabo el siguiente proceso:
1.	AssembledModel.py. Script del proceso completo, desde la lectura del archivo .dat, hasta la clasificación, a partir de datos nunca antes vistos, tomando como base la matriz de datos training “/trn_tst/X.csv”, y el vector de datos “/trn_tst/Y.csv”, empleando un algoritmo de Support Vector Machine.
  *	custom_features.json. Script que permite obtener los atributos en el dominio del tiempo.  Es llamado por AssembledModel.py

Toma como entrada el archivo un archivo nuevo .dat, y genera como salida, la clasificación en pantalla del tipo de movimiento introducido al algoritmo, de manera opcional, genera los heatmaps y gráficos de Amplitudes y PCA si se le indica que los guarde, al inicio de la ejecución del algoritmo. Utiliza la matriz de training X y el Vector de Clases Y, porque a partir de ellos se crean los respectivos modelos.

EJECUTANDO "AssembledModel.py":

1. Al inicio de la ejecución del software, se le preguntará si desea tratar los datos como matriz de testing o no:
 * Si se teclea SI, los archivos .dat que se seleccionen mediante la interfaz de ventanas del software, serán procesados y convertidos en una nueva matriz de datos X y vector de clases Y, y serán almacenados en la carpeta "/datos_nuevos", como "X_test.csv" y "Y_test.csv". La funcionalidad de ambos archivos es ser pasados como parámetros al script "EnsambleClasificadores.py", junto con la matriz de training "/trn_tst/X.csv" y el vector de training /trn_tst/Y.csv, para estimar la tasa de reconocimiento de cada Modelo construido al momento, por cada algoritmo de Clasificación (ver punto del proceso anterior).
NOTA: Esta opción puede ser utilizada para generar una nueva matriz de datos X y vector de clases Y para training, solo deben renombrarse y colocarse en la carpeta "/trn_tst".

![Rx03](https://user-images.githubusercontent.com/41920284/122335289-df12f800-ceef-11eb-85c3-65a74033685c.png)

 * Si se teclea NO, cada uno de los archivos .dat serán procesados y clasificados, y al final el software enviará dos mensajes de alerta sobre el tipo de movimiento detectado: Uno a través de Consola y otro a través de una ventana pop up.

2. Se pregunta al usuario si desea almacenar las imágenes de las señales (los heatmaps de Amplitud y subcarriers). Si se selecciona SI, los gráficos serán generados y almacenados en la carpeta "/images" 

![Rx08](https://user-images.githubusercontent.com/41920284/122335344-fa7e0300-ceef-11eb-9014-1610e15e6e5b.png)
![Rx09](https://user-images.githubusercontent.com/41920284/122335360-feaa2080-ceef-11eb-9d49-9e1cdc0c5d77.png)

3. Selección de archivos .dat. Después de los pasos anteriores, se abrirá una interfaz de ventanas donde el usuario podrá seleccionar los archivos .dat de los cuáles se desea determinar el tipo de movimiento al que pertenecen.
4. Finalmente, el Software envía las alertas generadas: Una a través de Consola y una más a través de una ventana pop up.

![Rx05](https://user-images.githubusercontent.com/41920284/122335568-58124f80-cef0-11eb-82eb-864be165ec33.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/41920284/122335575-5ba5d680-cef0-11eb-8c5a-23fb9de5bdc1.png">
</p>

NOTA: Si se seleccionan varios archivos .dat a la vez, serán procesados uno por uno, y no todos a la vez.
PAPER LINK: https://ieeexplore.ieee.org/document/9534823




