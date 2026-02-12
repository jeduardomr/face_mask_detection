import torch
import os

class Config:
    # ========== RUTAS DEL SISTEMA ==========
    DATA_DIR = "data/face-mask-data"
    
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
    MODEL_SAVE_PATH = "models/best_face_mask_model.pth"
    
    # ========== HIPERPARAMETROS DE ENTRENAMIENTO ==========
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15
    IMAGE_SIZE = 224
    
    # ========== CLASES A DETECTAR (EN INGLES - como vienen en el dataset) ==========
    CLASS_NAMES = [
        'with_mask',              # 0
        'without_mask',           # 1
        'mask_weared_incorrect',  # 2
        'gloves',                 # 3
        'cap',                    # 4
        'no_gloves',             # 5
        'no_cap'                 # 6
    ]
    
    # ========== NOMBRES EN ESPAÑOL PARA MOSTRAR AL USUARIO ==========
    # IMPORTANTE: Esto traduce las clases del ingles al español
    CLASS_DISPLAY_NAMES = {
        'with_mask': 'Mascarilla Puesta',
        'without_mask': 'Mascarilla Sin Poner', 
        'mask_weared_incorrect': 'Mascarilla Mal Puesta',
        'gloves': 'Guantes Puestos',
        'no_gloves': 'Sin Guantes',
        'cap': 'Cofia Puesta',
        'no_cap': 'Sin Cofia',
        'person': 'Persona'
    }
    
    # ========== COLORES PARA VISUALIZACION ==========
    CLASS_COLORS = {
        'with_mask': (0, 255, 0),        # Verde - Mascarilla correcta
        'without_mask': (0, 0, 255),     # Rojo - Sin mascarilla
        'mask_weared_incorrect': (0, 165, 255),  # Naranja - Mascarilla mal puesta
        'gloves': (255, 255, 0),         # Cyan
        'no_gloves': (128, 0, 128),      # Púrpura
        'cap': (255, 0, 255),            # Magenta
        'no_cap': (0, 255, 255),
        'person': (128, 128, 128)          # Amarillo
    }
    
    # ========== MENSAJES DE ALERTA EN ESPAÑOL ==========
    ALERT_MESSAGES = {
        'with_mask': 'Mascarilla correctamente puesta',
        'without_mask': 'ALERTA: Persona sin mascarilla',
        'mask_weared_incorrect': 'ALERTA: Mascarilla mal puesta',
        'gloves': 'Guantes detectados',
        'no_gloves': 'ALERTA: Sin guantes',
        'cap': 'Cofia detectada',
        'no_cap': 'ALERTA: Sin cofia'
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("training", exist_ok=True)
        os.makedirs("detection", exist_ok=True)