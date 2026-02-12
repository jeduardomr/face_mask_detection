import sys
import os
from pathlib import Path

# ========== AGREGAR RAÍZ DEL PROYECTO AL PATH ==========
# Esto permite importar utils.config aunque ejecutes desde cualquier subcarpeta
current_file = Path(__file__).resolve()          # ruta absoluta a este script
project_root = current_file.parent.parent        # training/.. = raíz del proyecto
sys.path.insert(0, str(project_root))            # añadir al inicio del path

from ultralytics import YOLO
from utils.config import Config

def train_yolo():
    """Entrena YOLO11n con el dataset y clases definidas en Config.CLASS_NAMES"""
    
    # ========== 1. RAÍZ DEL PROYECTO (YA ESTÁ EN project_root) ==========
    print(f"📁 Directorio raíz del proyecto: {project_root}")

    # ========== 2. VERIFICAR ESTRUCTURA DE DATOS ==========
    data_dir = project_root / "data" / "face-mask-data"
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"❌ No se encuentra el directorio de imágenes: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"❌ No se encuentra el directorio de etiquetas: {labels_dir}")

    num_images = len(list(images_dir.glob('*.*')))
    num_labels = len(list(labels_dir.glob('*.txt')))
    print(f"✅ Imágenes encontradas: {num_images}")
    print(f"✅ Etiquetas YOLO encontradas: {num_labels}")

    if num_images == 0 or num_labels == 0:
        raise RuntimeError("El dataset está vacío. Ejecuta primero convert_voc_to_yolo.py")

    # ========== 3. CREAR ARCHIVO YAML CON RUTA ABSOLUTA ==========
    (project_root / "training").mkdir(exist_ok=True)
    yaml_path = project_root / "training" / "face_mask_dataset.yaml"
    
    nc = len(Config.CLASS_NAMES)
    names = Config.CLASS_NAMES

    with open(yaml_path, "w") as f:
        f.write(f"""# Dataset generado automáticamente por train_yolo.py
    path: {data_dir.as_posix()}
    train: images
    val: images
    nc: {nc}
    names: {names}
    """)
    print(f"✅ Archivo YAML creado con {nc} clases: {names}")

    # ========== 4. CARGAR MODELO Y ENTRENAR ==========
    model = YOLO("yolo11n.pt")
    print("🚀 Modelo YOLO11n cargado. Comenzando entrenamiento...")

    runs_dir = project_root / "runs" / "detect"

    results = model.train(
        data=yaml_path.as_posix(),
        epochs=50,
        imgsz=640,
        batch=16,           # reduce a 8 si tienes poca memoria
        device=0,           # 'cpu' si no tienes GPU
        patience=10,
        save_period=5,
        project=runs_dir.as_posix(),
        name="yolo11n_facemask",
        exist_ok=True,
        hsv_h=0.015,   # variación de tono
        hsv_s=0.7,     # saturación
        hsv_v=0.4,     # valor
        degrees=10.0,  # rotación
        translate=0.1, # traslación
        scale=0.5,     # escala
        shear=2.0,     # cizallamiento
        flipud=0.5,    # volteo vertical
        fliplr=0.5,    # volteo horizontal
        mosaic=1.0    # mosaico (mezcla 4 imágenes)
    )

    print("🎉 Entrenamiento completado exitosamente.")
    print(f"💾 Modelo guardado en: {runs_dir / 'yolo11n_facemask' / 'weights' / 'best.pt'}")
    
    return results

if __name__ == "__main__":
    train_yolo()