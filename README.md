# Sistema de Detección de EPP en Tiempo Real

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO11n-00CCFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Licencia](https://img.shields.io/badge/licencia-AGPL--3.0-blue.svg)](LICENSE)

Detección en tiempo real de **mascarillas, guantes y cofias** con YOLO11n, e identificación de cuándo no se utilizan. Pensado para control de acceso, entornos sanitarios, laboratorios e industria.

---

## Contenido

- [Cómo funciona](#cómo-funciona)
- [Clases detectadas](#clases-detectadas)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Dataset](#dataset)
- [Entrenamiento](#entrenamiento)
- [Detección en tiempo real](#detección-en-tiempo-real)
- [Configuración](#configuración)
- [Docker y devcontainer](#docker-y-devcontainer)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Verificación](#verificación)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Licencia](#licencia)

---

## Cómo funciona

Corren dos modelos YOLO11n en el mismo bucle de fotogramas:

| Modelo | Peso | Función | Clases |
|--------|------|---------|--------|
| `model_person` | `yolo11n.pt` (COCO) | Detección de personas + tracking ByteTrack | 1 (persona) |
| `model_ppe` | `runs/detect/yolo11n_facemask/weights/best.pt` | Detección de elementos de EPP | 7 |

Por fotograma:

1. `model_person.track(classes=[0], persist=True)` asigna un `track_id` estable a cada persona (ByteTrack, `conf ≥ 0.5`).
2. `model_ppe` detecta EPP (`conf ≥ 0.3`). Cada detección se asigna a la persona más cercana por distancia entre centros, con un radio máximo de 200 px.
3. Una persona sin detección asociada de un elemento se registra como estado **negativo** (`without_mask`, `no_gloves`, `no_cap`).
4. Los contadores reflejan el **estado actual** de cada persona: suben cuando alguien entra o cambia de estado, y bajan cuando el track desaparece. Las alertas de audio tienen un cooldown independiente de 5 s por tipo.

> **Nota**: el conteo es instantáneo, no acumula histórico.

## Clases detectadas

El orden se define una única vez en `utils/config.py` → `Config.CLASS_NAMES`. **No reordenar**: los ids del modelo entrenado dependen de él.

| ID | Clave | Mostrado en pantalla | Color BGR |
|----|-------|----------------------|-----------|
| 0 | `with_mask` | Mascarilla Puesta | `(0, 255, 0)` verde |
| 1 | `without_mask` | Mascarilla Sin Poner | `(0, 0, 255)` rojo |
| 2 | `mask_weared_incorrect` | Mascarilla Mal Puesta | `(0, 165, 255)` naranja |
| 3 | `gloves` | Guantes Puestos | `(255, 255, 0)` cian |
| 4 | `cap` | Cofia Puesta | `(255, 0, 255)` magenta |
| 5 | `no_gloves` | Sin Guantes | `(128, 0, 128)` morado |
| 6 | `no_cap` | Sin Cofia | `(0, 255, 255)` amarillo |

Texto en pantalla y colores se editan en `CLASS_DISPLAY_NAMES` y `CLASS_COLORS`.

## Requisitos

- Python 3.10 – 3.12 (`torch==2.3.0` no publica wheels para 3.13+).
- Cámara web en el índice 0.
- GPU NVIDIA con CUDA: **opcional**, acelera el entrenamiento.

## Instalación

```bash
git clone <url-del-repositorio>
cd face_mask_detection

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset

Estructura esperada bajo `data/face-mask-data/`:

```text
data/face-mask-data/
├── images/    # .jpg / .jpeg / .png
└── labels/    # un .txt por imagen, formato YOLO
```

Cada línea de etiqueta: `<class_id> <x_center> <y_center> <width> <height>`, normalizado a 0–1.

El dataset no se versiona ni se reconstruye desde el repo: son fotos de personas
reales, fusionadas aparte. Se espera ya colocado en `data/face-mask-data/`.

## Entrenamiento

```bash
python training/train_yolo.py
```

`main.py` hace lo mismo automáticamente: si no encuentra `runs/detect/yolo11n_facemask/weights/best.pt`, dispara el entrenamiento antes de abrir la cámara.

| Parámetro | Valor | Nota |
|-----------|-------|------|
| `epochs` | 50 | early stop con `patience=10` |
| `imgsz` | 640 | |
| `batch` | 16 | bajar a 8 con poca VRAM |
| `device` | `auto` | ver `TRAIN_DEVICE` abajo |
| `save_period` | 5 | checkpoint cada 5 épocas |
| augmentation | — | HSV, rotación, escala, shear, flip, mosaic |

**Split train/val.** Antes de entrenar se copia el dataset a `runs/split/{train,val}` con un 80/20 (`seed=42`) y se regenera `training/face_mask_dataset.yaml` apuntando ahí con la ruta absoluta de tu máquina. Ese YAML es un artefacto generado: no se versiona y no debe editarse a mano. Para empezar de cero tras cambiar el dataset, borra `runs/split/`.

**Device.** `utils/config.py` lee la variable de entorno `TRAIN_DEVICE`: `auto` (por defecto, usa CUDA si está disponible), `cpu`, o `0` para forzar la GPU 0.

```bash
TRAIN_DEVICE=cpu python training/train_yolo.py
```

## Detección en tiempo real

```bash
python main.py
```

| Tecla | Acción |
|-------|--------|
| `q` | Salir |

En pantalla: bounding boxes con color y confianza por clase, `ID:<track_id>` por persona, avisos `Sin Guantes` / `Sin Cofia` bajo quien no los lleva, y el panel de estadísticas. El panel muestra el estado actual de cada elemento, `EPI completo`, y `Tracks unicos`, que es acumulado: no baja, y cuenta como track nuevo cada vez que ByteTrack re-adquiere a alguien tras una oclusión. El beep de alerta suena con `pygame`; si no hay dispositivo de audio el sistema degrada a avisos visuales sin fallar.

## Configuración

Todo en `utils/config.py`:

| Constante | Por defecto | Para qué |
|-----------|-------------|----------|
| `CLASS_NAMES` | 7 claves | orden de clases — no cambiar tras entrenar |
| `CLASS_DISPLAY_NAMES` | español | texto en pantalla |
| `CLASS_COLORS` | BGR | color de cada bounding box |
| `ALERT_MESSAGES` | español | texto de la alerta de audio |
| `CAMERA_INDEX` | `0` | índice del dispositivo |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | resolución de captura |
| `PERSON_CONF_THRESH` | `0.5` | confianza mínima para aceptar un track |
| `PPE_CONF_THRESH` | `0.3` | confianza mínima para un elemento de EPP |
| `ASSOCIATION_DIST_THRESH` | `200` | radio en px para asociar EPP a una persona |
| `AUDIO_COOLDOWN` | `5` | segundos entre alertas del mismo tipo |
| `TARGET_FPS` | `15` | techo de fotogramas por segundo |
| `TRACK_MAX_AGE` | `30` | fotogramas sin ver un track antes de borrarlo |
| `DEVICE_TRAIN` | `$TRAIN_DEVICE` | device del entrenamiento |

## Docker y devcontainer

```bash
xhost +local:root       # solo una vez, para que OpenCV abra la ventana
docker compose up --build
```

La imagen es `python:3.10-slim` con **torch CPU** (el contenedor no expone la GPU). El compose mapea `/dev/video0`, monta `/tmp/.X11-unix` en solo lectura, `data/face-mask-data/`, `runs/` y `models/`, fija `SDL_AUDIODRIVER=dummy` para que los beeps no fallen sin audio, `no-new-privileges:true` y `restart: unless-stopped` para que un crash no deje el puesto sin vigilancia. No la expongas a una LAN: con `network_mode: host` alcanza cualquier servicio local.

`.devcontainer/` reutiliza ese mismo compose con un override mínimo que mantiene el contenedor vivo y monta el código en `/app`. Al crearlo ejecuta `post-create.sh`, que solo verifica entorno (dependencias, dataset, pesos, cámara y X11) sin instalar nada. Todo corre como root: lo que se escriba en `runs/` o `data/` pertenece a root en el host.

## Estructura del proyecto

```text
face_mask_detection/
├── main.py                      # verifica dataset → entrena si falta → detecta
├── requirements.txt
├── Dockerfile                   # python:3.10-slim
├── docker-compose.yml           # cámara, X11, volúmenes y red del host
├── xhost_setup.sh               # comprueba que esté puesto el permiso X11 del contenedor
├── yolo11n.pt                   # pesos COCO base
├── detection/
│   └── real_time_detection.py   # bucle de doble modelo en vivo
├── training/
│   ├── train_yolo.py            # split + entrenamiento YOLO11n
│   └── face_mask_dataset.yaml   # GENERADO en tiempo de ejecución
├── utils/
│   ├── config.py                # clases, colores, cámara, umbrales
│   └── weights.py               # comprobación de integridad de los pesos
├── tests/
│   ├── run_all.sh               # toda la verificación (~3 s)
│   ├── test_constraints.py      # floor de calidad: comentarios, config, sintaxis
│   ├── test_state_machine.py    # invariantes de los contadores
│   └── test_split.py            # split 80/20 y limpieza
├── data/face-mask-data/         # dataset (ignorado por git)
├── runs/detect/yolo11n_facemask # pesos, results.csv y métricas PNG (versionados)
└── .devcontainer/               # entorno de desarrollo
```

## Verificación

No hay CI, pero hay tres checks que corren en local sin GPU, sin cámara y sin dataset:

```bash
bash tests/run_all.sh
```

Cubren el contrato de calidad, los invariantes de los contadores y el split train/val. Ejecútalos antes de dar cualquier cambio por bueno.

## Limitaciones conocidas

- **Sin split estratificado.** El 80/20 es un shuffle global, no por clase; con clases raras la validación puede quedar desbalanceada.
- **`torch==2.3.0` está en el rango de CVE-2025-32434** (ejecución de código al deserializar un `.pt` malicioso). Solo importa si cargas pesos de fuera; corregirlo exige subir a `torch>=2.6.0` y revisar la compatibilidad con Python.
- **El contenedor corre como root** con la red del host y el socket X11 montado. Está pensado para desarrollo local; no publiques la imagen ni le des acceso a una LAN.
- **Asignación por proximidad.** Un box de EPP se asigna a la persona geométricamente más cercana dentro del radio; con varios cuerpos solapados la asignación puede ser incorrecta.
- **El estado negativo es por defecto.** Si no se detecta una mascarilla, se cuenta como `without_mask`. Prioriza la seguridad, pero genera falsos positivos con detecciones parciales.
- **Validación en el mismo equipo.** Los hiperparámetros no están ajustados para este dominio; `results.csv` y las curvas de `runs/` son el punto de partida.
- **Entrenamiento lento en CPU.** 50 épocas en CPU tardan horas; con `device=0` es minutos.
- **Guantes oscuros.** El modelo tiende a fallar con guantes negros; añade ejemplos y reentrena.

## Licencia

GNU Affero General Public License v3.0. El texto completo está en [LICENSE](LICENSE). Copyright (c) 2026 jeduardomr.

El proyecto usa [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), que es AGPL-3.0, así que va con la misma licencia: no hubo elección. `yolo11n.pt`, incluido en el repo, también es de Ultralytics y viene con los mismos términos. Si esto se expone algún día como servicio web, la sección 13 pide que el código siga disponible para quien lo use.

El dataset no está en el repositorio y no se redistribuye: son caras de personas reales. Si traes otro, comprueba cómo lo licenció su autor.
