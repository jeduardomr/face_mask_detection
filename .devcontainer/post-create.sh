#!/usr/bin/env bash
# Sin `-e` a propósito: el script nunca falla la creación del contenedor.
set -uo pipefail

ok()   { printf 'OK  %s\n' "$1"; }
warn() { printf 'AVISO  %s\n' "$1"; }
info() { printf 'INFO  %s\n' "$1"; }

echo "═══════════════════════════════════════════════════════════"
echo " ENTORNO DE DESARROLLO — DETECCIÓN DE EPP (YOLO11n)"
echo "═══════════════════════════════════════════════════════════"

if python -c "import torch, ultralytics, cv2, pygame" >/dev/null 2>&1; then
    ok "Dependencias OK — $(python -c 'import torch, ultralytics; print(f"torch {torch.__version__} · ultralytics {ultralytics.__version__}")')"
else
    warn "Faltan dependencias de Python. Reconstruye la imagen: docker compose build"
fi
info "Intérprete: $(command -v python) · CUDA disponible: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'no')"

read -r num_images num_labels <<<"$(python - <<'PY'
from pathlib import Path
img = Path("data/face-mask-data/images")
lab = Path("data/face-mask-data/labels")
count = lambda p, s="*.*": len(list(p.glob(s))) if p.is_dir() else 0
print(count(img), count(lab, "*.txt"))
PY
)"
if [ "${num_images:-0}" -gt 0 ] && [ "${num_labels:-0}" -gt 0 ]; then
    ok "Dataset montado en data/face-mask-data/ — ${num_images} imágenes, ${num_labels} etiquetas"
else
    warn "Dataset vacío o no montado. Colócalo en ./data/face-mask-data/{images,labels} en el host."
fi

if [ -f "runs/detect/yolo11n_facemask/weights/best.pt" ]; then
    ok "Modelo EPP disponible — no hará falta reentrenar"
else
    info "Sin runs/detect/yolo11n_facemask/weights/best.pt: main.py entrenará (50 épocas, CPU)."
fi
[ -f "yolo11n.pt" ] || warn "Falta yolo11n.pt en la raíz; el entrenamiento no podrá cargar el modelo base."

if [ -e /dev/video0 ]; then
    ok "Cámara presente en /dev/video0 (índice Config.CAMERA_INDEX = 0)"
else
    warn "No existe /dev/video0. Añádelo al servicio epp-detector en docker-compose.yml."
fi
# El socket es X<n>, sin host ni número de pantalla: DISPLAY=:0.0 -> X0
x_display="${DISPLAY:-}"
x_display="${x_display#*:}"; x_display="${x_display%%.*}"
if [ -S "/tmp/.X11-unix/X${x_display}" ]; then
    ok "X11 disponible (DISPLAY=${DISPLAY}) — la ventana de la cámara se abrirá"
else
    warn "X11 no disponible. En el host ejecuta: xhost +local:root"
fi
info "Audio: SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-sin definir} (dummy = beeps desactivados)"

cat <<'EOF'

───────────────────────────────────────────────────────────
 Primeros pasos:
   python main.py                  pipeline completo (entrena si hace falta + cámara)
   python training/train_yolo.py   solo entrenamiento
───────────────────────────────────────────────────────────
EOF