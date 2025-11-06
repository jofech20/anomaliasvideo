import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import random

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
DATASET_PATH = "UCF_Crime_Frames/"
SAVE_MODEL_PATH = "cnn_lstm_model.keras"  # ✅ nuevo formato correcto
SEQ_LEN = 16
IMG_SIZE = (128, 128)
CHANNELS = 1
BATCH_SIZE = 8
EPOCHS = 10
MULTICLASS = False  # ✅ clasificación binaria (anómalo vs normal)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================
# FUNCIÓN PARA CARGAR SECUENCIAS DE FRAMES
# =====================================================
def load_sequences(base_path, seq_len=SEQ_LEN):
    sequences = []
    labels = []
    categories = sorted(os.listdir(base_path))
    print(f"→ Detectadas {len(categories)} categorías.")

    for idx, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        if not os.path.isdir(category_path):
            continue

        frames = sorted(os.listdir(category_path))
        for i in range(0, len(frames) - seq_len, seq_len):
            seq_frames = frames[i:i+seq_len]
            seq_paths = [os.path.join(category_path, f) for f in seq_frames]
            sequences.append(seq_paths)                                                                                                   

            # ✅ Etiqueta binaria: anomalía = 1, normal = 0
            label = 1 if category.lower() != "normal" else 0
            labels.append(label)

    return sequences, np.array(labels)

# =====================================================
# GENERADOR DE DATOS
# =====================================================
class FrameSequence(Sequence):
    def __init__(self, sequences, labels, batch_size=BATCH_SIZE, img_size=IMG_SIZE, channels=CHANNELS, shuffle=True):
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels
        self.shuffle = shuffle
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        return len(self.sequences) // self.batch_size

    def __getitem__(self, index):
        idxs = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(idxs)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, idxs):
        X = np.zeros((self.batch_size, SEQ_LEN, *self.img_size, self.channels), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.float32)

        for i, idx in enumerate(idxs):
            seq_paths = self.sequences[idx]
            frames = []
            for fp in seq_paths:
                img = tf.keras.preprocessing.image.load_img(fp, color_mode='grayscale', target_size=self.img_size)
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                frames.append(img)
            X[i] = np.array(frames)
            y[i] = self.labels[idx]

        return X, y

# =====================================================
# MODELO CNN + LSTM
# =====================================================
def build_cnn_lstm_model(input_shape=(SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    input_layer = layers.Input(shape=input_shape)

    # Bloques convolucionales
    x = layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu', padding='same'))(input_layer)
    x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(layers.Conv2D(128, (3,3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)

    # ✅ Salida binaria (una neurona)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=out)
    return model

# =====================================================
# CARGA DE DATOS
# =====================================================
print("→ Preparando secuencias (train)...")
train_path = os.path.join(DATASET_PATH, "train")
test_path = os.path.join(DATASET_PATH, "test")

train_sequences, train_labels = load_sequences(train_path)
test_sequences, test_labels = load_sequences(test_path)

print(f"Train sequences: {len(train_sequences)}, Test sequences: {len(test_sequences)}")

train_gen = FrameSequence(train_sequences, train_labels)
test_gen = FrameSequence(test_sequences, test_labels, shuffle=False)

# =====================================================
# COMPILACIÓN Y ENTRENAMIENTO
# =====================================================
model = build_cnn_lstm_model()
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

cbks = [
    callbacks.ModelCheckpoint(SAVE_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=cbks
)

# =====================================================
# GUARDAR MODELO FINAL
# =====================================================
model.save(SAVE_MODEL_PATH)
print(f"✅ Entrenamiento completado. Modelo guardado en: {SAVE_MODEL_PATH}")
