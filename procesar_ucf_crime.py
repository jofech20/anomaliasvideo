import os
import cv2
import glob
from tqdm import tqdm

# Ruta principal del dataset
DATASET_PATH = "UCF_Crime_Dataset/"
OUTPUT_PATH = "UCF_Crime_Frames/"
FRAME_SIZE = (128, 128)

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Recorre 'train' y 'test'
splits = ["train", "test"]

for split in splits:
    split_path = os.path.join(DATASET_PATH, split)
    output_split_path = os.path.join(OUTPUT_PATH, split)
    os.makedirs(output_split_path, exist_ok=True)

    # Listar categorías dentro del split
    categories = sorted(os.listdir(split_path))

    for category in categories:
        category_path = os.path.join(split_path, category)
        if not os.path.isdir(category_path):
            continue

        output_category_path = os.path.join(output_split_path, category)
        os.makedirs(output_category_path, exist_ok=True)

        # Listar todas las imágenes PNG dentro de la categoría
        image_files = sorted(glob.glob(os.path.join(category_path, "*.png")))

        print(f"\nProcesando {split}/{category} ({len(image_files)} imágenes)")

        for img_path in tqdm(image_files, desc=f"{split}/{category}", leave=False):
            # Leer la imagen
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] No se pudo leer: {img_path}")
                continue

            # Convertir a escala de grises y redimensionar
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, FRAME_SIZE)

            # Guardar imagen procesada en la carpeta de salida
            filename = os.path.basename(img_path)
            output_img_path = os.path.join(output_category_path, filename)
            cv2.imwrite(output_img_path, gray)

print("\n✅ Procesamiento completado. Frames redimensionados guardados en:", OUTPUT_PATH)
