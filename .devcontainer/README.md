# Devcontainer: Detector de EPP (YOLO11n)

El devcontainer reutiliza el `Dockerfile` y el `docker-compose.yml` de la raíz del
proyecto: la imagen y las dependencias son las mismas que en producción.

## Ficheros

| Fichero                              | Función                                                                 |
|--------------------------------------|------------------------------------------------------------------------|
| `devcontainer.json`                  | Configuración de VS Code / devcontainer CLI.                            |
| `docker-compose.devcontainer.yml`    | Override: mantiene el contenedor vivo y monta el código en `/app`.       |
| `post-create.sh`                     | Comprobaciones del entorno al crear el contenedor (no instala nada).    |

## Requisitos en el host

```bash
xhost +local:root          # permisos de X11 para la ventana de la cámara
```

Además:

- Docker + Docker Compose v2.
- Dataset en `data/face-mask-data/{images,labels}` (se monta por volumen).
- `DISPLAY` exportado en el shell desde el que se abre VS Code.

## Uso

1. Abre la carpeta del proyecto en VS Code.
2. *Reopen in Container* (o "Dev Containers: Reopen in Container").
3. La primera vez construye la imagen; a partir de ahí reutiliza la capa de cache.

Desde la terminal del devcontainer:

```bash
python main.py                  # pipeline completo: verifica dataset → entrena si falta → cámara
python training/train_yolo.py   # solo entrenamiento (CPU: puede tardar bastante)
```

## Notas

- Todo se ejecuta como root (la imagen no define usuario). Lo que se escriba en
  los volúmenes `runs/` o `data/` pertenece a root en el host.
- Sin reenvío de puertos: `network_mode: host` lo impide y no hace falta
  (el único servicio expuesto es la ventana de OpenCV vía X11).
- Entrenamiento en CPU: el devcontainer fija `TRAIN_DEVICE=cpu`
  (lo lee `utils/config.py::DEVICE_TRAIN`). Entrenar 50 épocas en CPU es lento;
  para entrenar con GPU es más rápido hacerlo en el host con `device=0`.
- Audio desactivado: `SDL_AUDIODRIVER=dummy` evita que `pygame.mixer` falle en
  el contenedor; los beeps no se oyen.