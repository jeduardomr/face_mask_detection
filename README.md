# Sistema de Detección de EPP en Tiempo Real

Sistema de visión por computadora basado en **YOLO11n** para detectar el uso de **mascarillas, guantes y cofias** en tiempo real, además de identificar cuándo **no se están utilizando**. Diseñado para control de acceso, entornos sanitarios, laboratorios e industria.

---

## 🚀 Características

- **7 clases detectadas**:
  - `with_mask` – mascarilla correcta
  - `without_mask` – sin mascarilla
  - `mask_weared_incorrect` – mascarilla mal puesta
  - `gloves` – guantes puestos
  - `no_gloves` – sin guantes
  - `cap` – cofia puesta
  - `no_cap` – sin cofia

- **Detección simultánea** de personas (tracking con ByteTrack) y elementos de EPP.
- **Estadísticas dinámicas**: los contadores se actualizan cuando una persona cambia su estado.
- **Alertas visuales y de audio** al detectar falta de EPP.
- **Entrenamiento automático** desde el script principal.
- **Configuración centralizada** en `utils/config.py`.

---

## 🧠 Modelo

- **Backbone**: `YOLO11n` (versión nano de Ultralytics YOLO11).
- **Dos modelos en paralelo**:
  - `model_person`: YOLO11n base (clase `person` de COCO) → tracking.
  - `model_ppe`: YOLO11n fine‑tuneado con las 7 clases → detección de EPP.

---

## 📦 Requisitos

- Python 3.10 – 3.12
- Cámara web
- (Opcional) GPU NVIDIA con CUDA para entrenamiento rápido

---

## 🔧 Instalación

1. **Clonar el repositorio**  
   ```bash
   git clone <url-del-repositorio>
   cd face_mask_detection
2. **Crear y activar entorno virtual**  
   ```bash
   python -m venv venv
   source venv/bin/activate
3. **Instalar dependencias**  
   ```bash
   pip install -r requirements.txt

## Orden de clases (ID = índice en esta lista):

    python
    ['with_mask', 'without_mask', 'mask_weared_incorrect', 'gloves', 'cap', 'no_gloves', 'no_cap']
    (Definido en utils.config.CLASS_NAMES)

    Cada línea del archivo .txt debe tener el formato:

    text
    <class_id> <x_center> <y_center> <width> <height>
    Coordenadas normalizadas entre 0 y 1.

## 🏋️ Entrenamiento
    Automático (recomendado)
    Ejecuta main.py. Si no existe el modelo runs/detect/yolo11n_facemask/weights/best.pt, se iniciará el entrenamiento  automáticamente.

    Manual
    bash
    python training/train_yolo.py
    Puedes ajustar hiperparámetros (épocas, batch, etc.) directamente en el script.

    El modelo entrenado se guarda en:
    runs/detect/yolo11n_facemask/weights/best.pt

## 🎥 Detección en tiempo real
    bash
    python main.py
    Controles:

    q → salir

    r → reiniciar estadísticas

    c → guardar captura de pantalla (screenshot_<timestamp>.jpg)

    En pantalla se muestran:

    Bounding boxes con colores por clase (nombres traducidos al español).

    Iconos de estado debajo de cada persona:
    😷 / 😨 / 😕 – estado de mascarilla
    🧤 / ✋ – guantes / sin guantes
    ⛑️ / 👤 – cofia / sin cofia

    Texto de ausencia: ❌ Sin guantes / ❌ Sin cofia cuando corresponde.

    Estadísticas acumuladas: personas únicas, conteo por clase, EPI completo.

## ⚙️ Configuración
    Todo se configura desde utils/config.py:

    CLASS_NAMES – orden de las clases (no cambiar si ya entrenaste).

    CLASS_DISPLAY_NAMES – traducciones al español para mostrar en pantalla.

    CLASS_COLORS – colores BGR para bounding boxes.

    ALERT_MESSAGES – mensajes de alerta (audio/consola).

    DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, etc.

## 📁 Estructura del proyecto (simplificada)
    text
    face_mask_detection/
    ├── main.py                     # Punto de entrada
    ├── training/
    │   └── train_yolo.py          # Entrenamiento
    ├── detection/
    │   └── real_time_detection.py # Detección en vivo
    ├── utils/
    │   └── config.py              # Configuración global
    ├── data/
    │   └── face-mask-data/        # Dataset (imágenes y etiquetas)
    └── runs/                      # Modelos entrenados (generado)
    
## 📌 Notas
    Alertas de audio: en Windows usa winsound.Beep(). En Linux/macOS se puede adaptar con pygame.

    Guantes negros: si el modelo no los detecta bien, añade más ejemplos de guantes negros a tu dataset y reentrena.

    Evaluación del modelo: desde Python puedes ejecutar:

    python
    from ultralytics import YOLO
    model = YOLO("runs/detect/yolo11n_facemask/weights/best.pt")
    model.val()
