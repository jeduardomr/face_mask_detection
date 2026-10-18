import os
from typing import List, Dict, Tuple

class Config:
    # Claves en ingles a proposito (son las del dataset). La posicion es el id de
    # clase con el que se entreno el modelo: no reordenar.
    CLASS_NAMES: List[str] = [
        'with_mask',
        'without_mask',
        'mask_weared_incorrect',
        'gloves',
        'cap',
        'no_gloves',
        'no_cap'
    ]

    CLASS_DISPLAY_NAMES: Dict[str, str] = {
        'with_mask': 'Mascarilla Puesta',
        'without_mask': 'Mascarilla Sin Poner',
        'mask_weared_incorrect': 'Mascarilla Mal Puesta',
        'gloves': 'Guantes Puestos',
        'no_gloves': 'Sin Guantes',
        'cap': 'Cofia Puesta',
        'no_cap': 'Sin Cofia',
        'person': 'Persona'
    }

    # Tuplas BGR de OpenCV, no RGB.
    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
        'with_mask': (0, 255, 0),               # Verde
        'without_mask': (0, 0, 255),            # Rojo
        'mask_weared_incorrect': (0, 165, 255), # Naranja
        'gloves': (255, 255, 0),                # Cyan
        'no_gloves': (128, 0, 128),             # Púrpura
        'cap': (255, 0, 255),                   # Magenta
        'no_cap': (0, 255, 255),                # Amarillo
        'person': (128, 128, 128)               # Gris
    }
    
    ALERT_MESSAGES: Dict[str, str] = {
        'without_mask': 'Persona sin mascarilla',
        'mask_weared_incorrect': 'Mascarilla mal puesta',
        'no_gloves': 'Sin guantes',
        'no_cap': 'Sin cofia'
    }
    
    DEVICE_TRAIN: str = os.environ.get('TRAIN_DEVICE', 'auto')
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480

    PERSON_CONF_THRESH: float = 0.5
    PPE_CONF_THRESH: float = 0.3
    ASSOCIATION_DIST_THRESH: int = 200   # px
    AUDIO_COOLDOWN: int = 5              # segundos
    TARGET_FPS: int = 15
    TRACK_MAX_AGE: int = 30              # frames
