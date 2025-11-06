import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ================================
# 1. Cargar dataset (frames)
# ================================
def cargar_dataset(path="processed_ucsd"):
    import os
    X, y = [], []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                frames = np.load(os.path.join(root, file))

                # Etiqueta: 0 = normal, 1 = anómalo
                label = 0 if "normal" in root.lower() else 1

                for frame in frames:  # <-- usar cada frame individual
                    X.append(frame)
                    y.append(label)

    return np.array(X), np.array(y)

X, y = cargar_dataset("processed_ucsd")
print("Dataset:", X.shape, y.shape)

# Normalizar por seguridad
X = X.astype("float32") / 255.0

# ================================
# 2. Dividir dataset
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# ================================
# 3. Definir modelo CNN
# ================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binario: normal vs anómalo
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ================================
# 4. Entrenar modelo
# ================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

# ================================
# 5. Evaluar modelo
# ================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n✅ Accuracy en test: {test_acc*100:.2f}%")
