import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
import seaborn as sns
import json
import csv

# ============================
# 1. Cargar datos y modelo
# ============================
print("Cargando datos y modelo...")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
model = tf.keras.models.load_model("cnn_lstm_model.keras")

print(f"Datos cargados: X_test={X_test.shape}, y_test={y_test.shape}")

# ============================
# 2. Predicciones del modelo
# ============================
print("Generando predicciones...")
y_pred_prob = model.predict(X_test, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int)

# ============================
# 3. Resultados de predicción
# ============================
plt.figure(figsize=(10, 4))
plt.plot(y_test[:100], 'b-', label='Real')
plt.plot(y_pred[:100], 'r--', label='Predicción')
plt.title("Resultados de predicción (100 muestras)")
plt.xlabel("Muestra")
plt.ylabel("Clase")
plt.legend()
plt.tight_layout()
plt.savefig("resultados_prediccion.png")
plt.close()
print("✅ Guardado: resultados_prediccion.png")

# ============================
# 4. Matriz de confusión
# ============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
plt.close()
print("✅ Guardado: matriz_confusion.png")

# ============================
# 5. Curva Precision-Recall
# ============================
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
ap = average_precision_score(y_test, y_pred_prob)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color="darkorange", lw=2,
         label=f"Curva PR (AP = {ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.legend()
plt.tight_layout()
plt.savefig("curva_precision_recall.png")
plt.close()
print("✅ Guardado: curva_precision_recall.png")

# ============================
# 6. Métricas y resumen
# ============================
reporte = classification_report(y_test, y_pred, target_names=["Normal", "Anómalo"], output_dict=True)

# Guardar TXT
with open("resumen_metricas.txt", "w") as f:
    f.write("=== RESUMEN DE MÉTRICAS ===\n\n")
    f.write(json.dumps(reporte, indent=4))

# Guardar JSON
with open("resumen_metricas.json", "w") as f_json:
    json.dump(reporte, f_json, indent=4)

# Guardar CSV
with open("resumen_metricas.csv", "w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["Clase", "Precision", "Recall", "F1-score", "Soporte"])
    for clase, valores in reporte.items():
        if isinstance(valores, dict):
            writer.writerow([
                clase,
                valores.get("precision", ""),
                valores.get("recall", ""),
                valores.get("f1-score", ""),
                valores.get("support", "")
            ])

print("✅ Guardado: resumen_metricas.txt, resumen_metricas.json, resumen_metricas.csv")
print("\n🎯 Evaluación completada correctamente.")


