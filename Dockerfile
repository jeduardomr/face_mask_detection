# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# libsm6/libice6 los exige el plugin Qt xcb de OpenCV: sin ellos cv2.imshow aborta.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libice6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 600 --retries 10 -r requirements.txt

COPY . .

RUN mkdir -p /app/data/face-mask-data/images /app/data/face-mask-data/labels /app/runs

CMD ["python", "main.py"]