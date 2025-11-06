# Proyecto: Detección de Anomalías en Videovigilancia con Deep Learning

Autores: Josué Hurtado, César Rivas
Descripción: Sistema basado en CNN+LSTM que detecta comportamientos anómalos en videos de vigilancia en tiempo real.

Requisitos:
- Python 3.10
- TensorFlow 2.12
- OpenCV, NumPy, tqdm

Ejecutar:
1. python procesar_ucf_crime.py  (extrae frames de los videos)
2. python train_cnn_lstm.py      (entrena la red y genera métricas)
3. Revisar resultados en carpeta "evidencias/"

Salida esperada:
- Modelo entrenado (.keras)
- Logs de entrenamiento
- Métricas de rendimiento
