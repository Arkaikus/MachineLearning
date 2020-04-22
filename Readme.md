# Paper Secuestros

## Objetivos

- Tratamiento de datos de Secuestros/Desapariciones de la Fiscalia en formato CSV
- Encontrar información mediante kmeans o redes neuronales o rn convolucionales
- Datos importantes: #Secuestros, #Desapariciones forzadas /por año /por departamento /# de condenados /denuncias

## Ideas

- Estudio de los 5 departamentos con más secuestros y desapariciones
- Correlacionar secuestros y desapariciones
- Correlacionar secuestros y desapariciones con acuerdos de paz (datos: antes, durante, y después)
- Entrenar un modelo para determinar si un caso de secuestro terminó en condena **[HECHO]**


## Tecnologías

- Python + Tensorflow/NumPy/Pandas/Matplotlib/JupyterLab/sklearn

## Datos de la Fiscalía

[Conteo de Victimas](https://www.datos.gov.co/Justicia-y-Derecho/Conteo-de-V-ctimas/sft7-9im5)
> Contiene muchos registros, pero solo nos interesa aquellos cuyo delito sea Secuestro o Desaparición

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
