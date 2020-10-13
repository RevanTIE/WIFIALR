# CSIProject

El siguiente proyecto incorpora scripts para el manejo de datasets, resultado de obtención de señales CSI desde Tx hasta Rx.
* datos_crudos. Carpeta que almacena matrices, de los datos convertidos de .dat a .csv, como resultado de ejecutar el script "/librerias/Activity_datfile_to_csvfile.m".
* img. Imagenes relacionadas al proyecto.
* librerias. Scripts de Daniel Halperin, e Hirokazu Narui, de los cuales se realizaron modificaciones propias a "Activity_datfile_to_csvfile.m".
* /librerias/Activity_datfile_to_csvfile.m. Script para convertir los datos .dat en archivos .csv. Las matrices resultantes van para la carpeta "/datos_crudos".
* pca. Matrices como resultado de aplicar a los datos preprocesados, el script de "PCA_STFT_visualize.ipynb".
* preprocesados. Matrices como resultado de aplicar a los datos crudos, el script de "preprocesamiento.py".
* trn_tst. Matriz de las X, y el vector de clases Y, que serán divididas en: training, test y validation; se generan como resultado del script "generarMatrizX.py".
* csi_headers.csv son los encabezados tomados para los archivos preprocesados, son utilizados por el script "preprocesamiento.py".
* CSI_preprocesamiento.ipynb. Script experimental para el preprocesamiento de datos.
* generarMatrizX.py. Script que genera la matriz X, y las clases Y; los archivos generados son enviados a "/trn_tst".
* PCA_STFT_visualize.ipynb. Script para extracción de los Principales Componentes, y generación de gráficas; los archivos generados son enviados a "/pca".
* preprocesamiento.py. Script para generar matrices de datos preprocesados, se almacenan en la carpeta "/preprocesados".

Pasos a seguir:
1. Una vez que los archivos .dat son generados por el dispositivo NUC RX, ejecutar el script "/librerias/Activity_datfile_to_csvfile.m" para cada uno de los movimientos. Como resultado se generan archivos .csv que van para la carpeta "/datos_crudos".
2. Ejecutar el script "preprocesamiento.py", para generar datos limpios. Como resultado se generan matrices de datos preprocesados de cada uno de los movimientos, dentro de la carpeta "/preprocesados".
3. Ejecutar el script "PCA_STFT_visualize.ipynb", para la generación de matrices de PCA, a partir de los datos preprocesados, así como gráficos dentro de Jupyter Notebook. Las matrices resultantes se envían a la carpeta "/pca".
4. Ejecutar el script "generarMatrizX.py" para generar la matriz X, y el vector de clases Y, que serán utilizados para el training, testing y validation.

Notas: "Activity_datfile_to_csvfile.m" se ejecuta con Matlab; los archivos "*.py" desde Spyder; y los archivos "*.ipynb" desde Jupyter Notebook.
