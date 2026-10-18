# Política de seguridad

## Versiones soportadas

| Versión | Soporte |
| --- | --- |
| `main` (último commit) | Sí |
| Commits anteriores, forks y copias locales | No |

El proyecto no publica releases ni mantiene ramas de soporte. Las correcciones
se aplican sobre `main`.

## Reportar una vulnerabilidad

**No abras un issue público** ni publiques una prueba de concepto antes de que
el fallo esté corregido.

Usa el reporte privado de GitHub:

[Reportar una vulnerabilidad](https://github.com/jeduardomr/face_mask_detection/security/advisories/new)

Incluye, en la medida de lo posible:

- Componente afectado (`detection/`, `training/`, `utils/`, Docker, scripts).
- Commit afectado (`git rev-parse --short HEAD`).
- Descripción del fallo y su impacto real.
- Pasos para reproducirlo con la mínima configuración necesaria.
- Entorno: SO, versión de Python, CPU/GPU, Docker o local.
- Parche o mitigación sugerida, si la tienes.

No adjuntes imágenes del dataset, capturas de cámara ni datos personales. Si el
fallo necesita una muestra, describe cómo generarla o usa datos sintéticos.

Puedes escribir en español o en inglés.

## Alcance

### Dentro del alcance

- `main.py`, `detection/`, `training/`, `utils/` y `tests/`.
- `Dockerfile`, `docker-compose.yml`, `.devcontainer/` y scripts de apoyo.
- Carga de pesos y validación de integridad (`utils/weights.py`).
- Procesamiento de rutas y ficheros al convertir o dividir datasets.
- Dependencias de `requirements.txt`, en lo que afecte a este proyecto.

### Fuera del alcance

- Vulnerabilidades en `ultralytics`, `torch`, `opencv-python` u otras
  dependencias: repórtalas al proyecto upstream. Dependabot ya las vigila aquí.
- El dataset de entrenamiento: no se distribuye con el repositorio.
- Despliegues locales mal configurados: contenedor como root,
  `network_mode: host`, `xhost +local:root` o X11 expuesto. El compose está
  pensado para un equipo local, no para una LAN o Internet.
- Ingeniería social, acceso físico y denegación de servicio sobre hardware
  propio.
- Informes teóricos sin impacto demostrable.

## Limitaciones de seguridad conocidas

- **CVE-2025-32434 (`torch==2.3.0`).** Permite ejecución de código al
  deserializar un `.pt` no confiable. Mitigación actual: los pesos se verifican
  contra `weights.sha256` antes de cargarse. Usa únicamente pesos de confianza.
  La corrección definitiva exige `torch>=2.6.0` y un cambio de baseline; ver
  `CONSTRAINTS.md` (D2).
- **Contenedor como root.** El `docker-compose.yml` monta la cámara y el socket
  X11, y usa la red del host. No lo expongas a una LAN ni publiques la imagen
  tal cual.
- **Sin servicio de red.** La aplicación no expone puertos ni autentica
  usuarios; el riesgo principal es local y depende de los ficheros que se le
  pasen.

## Divulgación coordinada

- Acuse de recibo orientativo: 7 días. No hay SLA ni bug bounty; es un proyecto
  personal.
- Objetivo: confirmar el fallo, preparar la corrección y publicar un aviso
  coordinado en un plazo de 90 días desde el reporte, o antes si hay parche.
- Si el fallo ya se explota públicamente, indícalo en el reporte para
  priorizarlo.
- Puedes solicitar crédito en el aviso; si no, se publicará sin atribución.

## Investigación de buena fe

No se tomarán acciones legales contra quien investigue de buena fe siguiendo
esta política. Limita las pruebas a repositorios y despliegues propios, no
accedas a datos de terceros y no degrade el servicio de nadie.

## Actualizaciones de seguridad

Las correcciones se aplican en `main`. No hay backports a commits anteriores.
Los avisos de dependencias se gestionan con Dependabot y, cuando aplica, se
documentan en `CONSTRAINTS.md`.

## Contacto

Usa exclusivamente el reporte privado de GitHub enlazado arriba. No se atenderán
vulnerabilidades reportadas mediante issues públicos, correo o redes sociales.
