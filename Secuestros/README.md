# Modelo de Redes Neuronales para la predicción de victimas de secuestro en Colombia

## Archivos

- ConteoVictimas(.*).csv: Datos filtrados de los datos abiertos de la fiscalía
- DataSet(.*): Datos con los que se entrena y prueban los modelos en los notebooks
- Analisis de Datos - Secuestros.ipynb : Clasificación de casos de secuestros por condena en colombia 
- Analisis de Datos - Tendencias (.*).ipynb : Regresión lineal con redes neuronales

## Objetivos

- Tratamiento de datos de Victimas/Indiciados de la Fiscalia en formato CSV **[HECHO]**
- Encontrar información mediante svm/redes neuronales **[NN Regresión]** **[HECHO]**
- Datos importantes: Victimas de secuestros por años y departamentos

## Tareas

- Top departamentos con más secuestros y desapariciones **[HECHO]**
- Correlacionar secuestros y desapariciones **[DESCARTADO]**
- Correlacionar secuestros y desapariciones con acuerdos de paz (datos: antes, durante, y después) **[DESCARTADO]**
- Entrenar un modelo para determinar si un caso de secuestro terminó en condena **[HECHO]**
- Estadisticas, mapa de calor departamentos, top departamentos en número de secuestros **[HECHO]**
- Revisar Paper Hurtos y reproducir gráficas **[HECHO]**
- Linea de regresión usando seaborn.regplot **[HECHO]** 
- Comparar svm y nn **[HECHO]**
- predecir a 10 años nn+svm **[HECHO]**

## Tecnologías

- Python + Tensorflow/NumPy/Pandas/Matplotlib/JupyterLab/sklearn/GeoPandas/Descartes/Seaborn/joblib

## Datos de la Fiscalía

[Conteo de Victimas](https://www.datos.gov.co/Justicia-y-Derecho/Conteo-de-V-ctimas/sft7-9im5)
> Contiene muchos registros, pero solo nos interesa aquellos cuyo delito sea Secuestro o Desaparición

## Mapa Colombia GeoJSON

https://bl.ocks.org/john-guerra/43c7656821069d00dcbc

### Filtrar Datos

Revisar utils.py

## Referencias

- [Keras](https://keras.io/)
- [Neural Networks Regression](https://missinglink.ai/guides/neural-network-concepts/neural-networks-regression-part-1-overkill-opportunity/)
- [Activation Functions](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/)
- [Neural Networks Colaboratory](https://colab.research.google.com/drive/1B1ZFXIPgDjKg7TQHhd7Nlmi5oz-N3CNo)
- [Datos Abiertos Fiscalía](https://www.datos.gov.co/browse?q=fiscalia%20spoa&sortBy=relevance)
- [Conteo de Victimas](https://www.datos.gov.co/Justicia-y-Derecho/Conteo-de-V-ctimas/sft7-9im5)
- [Conteo de Procesos](https://www.datos.gov.co/Justicia-y-Derecho/Conteo-de-Procesos/q6re-36rh)
- [Regresión Basica + Keras.Sequential](https://www.tensorflow.org/tutorials/keras/regression)
- [Classify Structured Data w/ Feature Columns](https://www.tensorflow.org/tutorials/structured_data/feature_columns)
- [Introducing TensorFlow Feature Columns](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)
- [Regression Keras](https://www.pluralsight.com/guides/regression-keras)
- [Map with GeoPandas](https://github.com/bendoesdata/make-a-map-geopandas)