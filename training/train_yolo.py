import sys
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Any

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from utils.config import Config
from utils.weights import verify_weight


def create_train_val_split(data_dir: Path, split_dir: Path, train_ratio: float = 0.8, seed: int = 42) -> Path:
    """Copia el dataset a split_dir con un split 80/20 por shuffle global.

    No estratifica: con clases raras la validacion puede quedar desbalanceada.
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    split_images_train = split_dir / "train" / "images"
    split_labels_train = split_dir / "train" / "labels"
    split_images_val = split_dir / "val" / "images"
    split_labels_val = split_dir / "val" / "labels"

    if split_dir.exists():
        # Sin ignore_errors: un borrado a medias mezclaria el split viejo con el
        # nuevo y las metricas de dos entrenamientos dejarian de ser comparables.
        shutil.rmtree(split_dir)
    for d in [split_images_train, split_labels_train, split_images_val, split_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    valid_pairs: List[Tuple[Path, Path]] = [
        (img_path, labels_dir / f"{img_path.stem}.txt")
        for img_path in images_dir.glob('*.*')
        if (labels_dir / f"{img_path.stem}.txt").exists()
    ]

    print(f"Total pares válidos: {len(valid_pairs)}")

    random.shuffle(valid_pairs)
    n_train = int(len(valid_pairs) * train_ratio)

    for pairs, img_dir, lbl_dir in ((valid_pairs[:n_train], split_images_train, split_labels_train),
                                    (valid_pairs[n_train:], split_images_val, split_labels_val)):
        for img_path, label_path in pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(label_path, lbl_dir / label_path.name)
        print(f"OK  {img_dir.parent.name}: {len(pairs)} pares")

    return split_dir


def train_yolo() -> Any:
    print(f"Directorio raíz del proyecto: {project_root}")

    data_dir = project_root / "data" / "face-mask-data"
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"ERROR  No se encuentra el directorio de imágenes: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"ERROR  No se encuentra el directorio de etiquetas: {labels_dir}")

    num_images = len(list(images_dir.glob('*.*')))
    num_labels = len(list(labels_dir.glob('*.txt')))
    print(f"OK  Imágenes encontradas: {num_images}")
    print(f"OK  Etiquetas YOLO encontradas: {num_labels}")

    if num_images == 0 or num_labels == 0:
        raise RuntimeError("El dataset está vacío. Comprueba data/face-mask-data/ antes de entrenar")

    split_dir = create_train_val_split(data_dir, project_root / "runs" / "split")

    (project_root / "training").mkdir(exist_ok=True)
    yaml_path = project_root / "training" / "face_mask_dataset.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# Generado por train_yolo.py — no editar a mano.
path: {split_dir.as_posix()}
train: train/images
val: val/images
nc: {len(Config.CLASS_NAMES)}
names: {Config.CLASS_NAMES}
""")
    print(f"OK  {yaml_path.name} creado con {len(Config.CLASS_NAMES)} clases")

    verify_weight("yolo11n.pt")
    model = YOLO("yolo11n.pt")

    print("Modelo YOLO11n cargado. Comenzando entrenamiento...")

    runs_dir = project_root / "runs" / "detect"

    device = Config.DEVICE_TRAIN
    if device == 'auto':
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"Device auto-detectado: {device}")
    else:
        print(f"Device configurado: {device}")

    results = model.train(
        data=yaml_path.as_posix(),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        patience=10,
        save_period=5,
        project=runs_dir.as_posix(),
        name="yolo11n_facemask",
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0
    )

    print("OK  Entrenamiento completado exitosamente.")
    print(f"Modelo guardado en: {runs_dir / 'yolo11n_facemask' / 'weights' / 'best.pt'}")

    return results


if __name__ == "__main__":
    train_yolo()
