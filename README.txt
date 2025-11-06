TÍTULO DEL PROYECTO
Aplicación de Técnicas de Deep Learning para la Detección de Anomalías en Sistemas de Videovigilancia en Tiempo Real – Lima 2025

DESCRIPCIÓN GENERAL

Este proyecto implementa un sistema basado en redes neuronales profundas (Deep Learning) para detectar comportamientos anómalos en videos de vigilancia.
Utiliza una arquitectura híbrida CNN-LSTM que combina redes convolucionales para extraer características espaciales de los frames y redes LSTM para analizar la secuencia temporal.

El modelo fue entrenado y evaluado utilizando el dataset público UCF-Crime, el cual contiene distintas categorías de eventos anómalos y normales (por ejemplo, Abuse, Robbery, Explosion, entre otros).

El propósito es contribuir a la detección temprana de incidentes mediante el análisis automatizado de video, simulando el funcionamiento de un sistema de videovigilancia inteligente.

ESTRUCTURA DEL PROYECTO

UCF_Crime_Dataset (Dataset original, con carpetas traintest y categorías)
│
├── data (Datos convertidos en formato numpy)
│ ├── X_train.npy
│ ├── X_test.npy
│ ├── y_train.npy
│ └── y_test.npy
│
├── modelos (Modelos entrenados y guardados)
│ └── cnn_lstm_model.keras
│
├── evidencias (Resultados y reportes de validación)
│ ├── log_entrenamiento.txt
│ ├── resultados_prediccion.png
│ ├── matriz_confusion.png
│ ├── curva_precision_recall.png
│ └── resumen_metricas.txt
│
├── scripts (Scripts del proyecto)
│ ├── convertir_dataset_ucfcrime.py → Convierte las imágenes del dataset a secuencias numpy
│ ├── train_cnn_lstm.py → Entrena el modelo CNN-LSTM
│ └── evaluate_cnn_lstm.py → Evalúa el modelo entrenado y genera métricas
│
└── README.txt (Este documento)

REQUISITOS DE INSTALACIÓN

Python 3.11 o 3.12 (recomendado)

Librerías necesarias

pip install tensorflow==2.20.0
pip install numpy==1.26.4
pip install opencv-python==4.9.0.80
pip install matplotlib seaborn scikit-learn

PROCESO DE USO

CONVERSIÓN DEL DATASET
Ejecuta el script para convertir las imágenes .png del dataset UCF-Crime a formato numpy (.npy)

python scriptsconvertir_dataset_ucfcrime.py

ENTRENAMIENTO DEL MODELO (opcional, si ya se tiene uno guardado)
python scriptstrain_cnn_lstm.py

EVALUACIÓN DEL MODELO
Ejecuta el archivo
python scriptsevaluate_cnn_lstm.py

Esto generará las métricas de desempeño (precisión, recall, f1-score), la matriz de confusión, la curva de precisión-recall y las predicciones gráficas.

RESULTADOS ESPERADOS

El modelo debe ser capaz de identificar comportamientos anómalos en videos simulando un entorno de vigilancia.
Los resultados de evaluación se almacenan en la carpeta “evidencias”, incluyendo las gráficas y los archivos de métricas.

AUTORES

Josué Felipe Hurtado Chávez
César Aarón Rivas Ramos
Escuela Profesional de Ingeniería de Sistemas Computacionales
Facultad de Ingeniería – Lima 2025

LICENCIA

Proyecto académico con fines de investigación y validación tecnológica.
Prohibida su distribución o uso comercial sin autorización de los autores.