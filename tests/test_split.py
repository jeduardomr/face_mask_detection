# python3 tests/test_split.py  (sin pytest, sin GPU, sin dataset real)
"""Prueba create_train_val_split: 80/20, limpieza previa y contenido."""
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules["ultralytics"] = types.ModuleType("ultralytics")
sys.modules["ultralytics"].YOLO = object

from training.train_yolo import create_train_val_split  # noqa: E402

tmp = Path(tempfile.gettempdir()) / "face_mask_split_fixture"
shutil.rmtree(tmp, ignore_errors=True)

data = tmp / "data"
images, labels = data / "images", data / "labels"
for d in (images, labels):
    d.mkdir(parents=True)

N = 100
for i in range(N):
    (images / f"img{i}.jpg").write_bytes(b"x")
    (labels / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
# una imagen sin etiqueta: no debe entrar en el split
(images / "sin_label.jpg").write_bytes(b"x")

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got}" + ("" if ok else f"  != {want}"))
    if not ok:
        failures.append(name)


split = create_train_val_split(data, tmp / "runs/split")
n_train = len(list((split / "train" / "images").glob("*.jpg")))
n_val = len(list((split / "val" / "images").glob("*.jpg")))

check("train 80%", n_train, 80)
check("val 20%", n_val, 20)
check("total excluye la imagen sin etiqueta", n_train + n_val, N)
check("labels train", len(list((split / "train" / "labels").glob("*.txt"))), n_train)
check("labels val", len(list((split / "val" / "labels").glob("*.txt"))), n_val)
check("sin_label fuera del split",
      list((split).rglob("sin_label.jpg")), [])

# Reejecucion: no debe duplicar ni dejar ficheros del run anterior.
(split / "train" / "images" / "fantasma.jpg").write_bytes(b"stale")
create_train_val_split(data, tmp / "runs/split")
check("limpia el destino previo",
      (split / "train" / "images" / "fantasma.jpg").exists(), False)
check("train sigue siendo 80 tras reejecutar",
      len(list((split / "train" / "images").glob("*.jpg"))), 80)

# Determinismo con la misma seed.
a = sorted(p.name for p in (split / "val" / "images").glob("*.jpg"))
create_train_val_split(data, tmp / "runs/split")
b = sorted(p.name for p in (split / "val" / "images").glob("*.jpg"))
check("misma seed -> mismo split", a, b)

shutil.rmtree(tmp, ignore_errors=True)
print()
print("TODO OK" if not failures else f"FALLARON: {failures}")
sys.exit(1 if failures else 0)