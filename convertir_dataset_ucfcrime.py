import os
import numpy as np
import cv2

BASE_PATH = r"C:\Users\dchur\OneDrive\Documentos\JOSUE\INVESTIGACION\UCF_Crime_Dataset"
SECUENCIAS = 16
IMG_SIZE = (128, 128)
BATCH_SIZE = 100  # número de secuencias antes de escribir al disco

def convertir_a_npy_stream(dataset_path, nombre_salida):
    clases = sorted(os.listdir(dataset_path))
    print(f"\n🔍 Clases detectadas en '{os.path.basename(dataset_path)}': {clases}")

    # Archivos temporales de salida
    X_out = f"X_{nombre_salida}.npy"
    y_out = f"y_{nombre_salida}.npy"

    # Borramos si ya existían
    if os.path.exists(X_out): os.remove(X_out)
    if os.path.exists(y_out): os.remove(y_out)

    X_buffer, y_buffer = [], []

    for clase_idx, clase in enumerate(clases):
        clase_path = os.path.join(dataset_path, clase)
        if not os.path.isdir(clase_path):
            continue

        imagenes = sorted([f for f in os.listdir(clase_path) if f.endswith('.png')])
        print(f"📂 Procesando clase '{clase}' ({len(imagenes)} imágenes)...")

        for i in range(0, len(imagenes) - SECUENCIAS + 1, SECUENCIAS):
            frames = []
            for j in range(SECUENCIAS):
                img_path = os.path.join(clase_path, imagenes[i + j])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype(np.float32) / 255.0
                frames.append(img)

            if len(frames) == SECUENCIAS:
                X_buffer.append(frames)
                y_buffer.append(clase_idx)

            # Guardar por bloques
            if len(X_buffer) >= BATCH_SIZE:
                X_np = np.array(X_buffer, dtype=np.float32).reshape(-1, SECUENCIAS, IMG_SIZE[0], IMG_SIZE[1], 1)
                y_np = np.array(y_buffer, dtype=np.int32)
                with open(X_out, 'ab') as f: np.save(f, X_np)
                with open(y_out, 'ab') as f: np.save(f, y_np)
                X_buffer, y_buffer = [], []

    # Guardar último bloque
    if X_buffer:
        X_np = np.array(X_buffer, dtype=np.float32).reshape(-1, SECUENCIAS, IMG_SIZE[0], IMG_SIZE[1], 1)
        y_np = np.array(y_buffer, dtype=np.int32)
        with open(X_out, 'ab') as f: np.save(f, X_np)
        with open(y_out, 'ab') as f: np.save(f, y_np)

    print(f"✅ Guardado incremental completado: {nombre_salida}")

def main():
    print("🚀 Iniciando conversión de UCF-Crime a formato .NPY (modo streaming)...\n")

    train_path = os.path.join(BASE_PATH, "Train")
    test_path = os.path.join(BASE_PATH, "Test")

    if os.path.exists(train_path):
        convertir_a_npy_stream(train_path, "train")
    if os.path.exists(test_path):
        convertir_a_npy_stream(test_path, "test")

    print("\n🎉 Conversión completada con éxito.")

if __name__ == "__main__":
    main()

