import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from ultralytics import YOLO
import pygame
from utils.config import Config
from utils.weights import verify_weight

# Estado de una persona -> clave del contador que lo agrega en Config stats.
_STAT_KEY = {
    'with_mask': 'with_mask',
    'without_mask': 'without_mask',
    'mask_weared_incorrect': 'incorrect_mask',
    'gloves': 'gloves',
    'no_gloves': 'no_gloves',
    'cap': 'cap',
    'no_cap': 'no_cap',
}
_ALERT_ON = frozenset({'without_mask', 'mask_weared_incorrect', 'no_gloves', 'no_cap'})
_PPE_FULL = {'mask': 'with_mask', 'gloves': 'gloves', 'cap': 'cap'}
_DEFAULT_NEGATIVE = {'mask': 'without_mask', 'gloves': 'no_gloves', 'cap': 'no_cap'}


def _assert_class_contract(names: Any) -> None:
    """El orden de Config.CLASS_NAMES es el contrato con el modelo entrenado.

    Si el .pt trae otra cosa, cada box se etiquetaria con la clase que le toca
    por indice y los contadores mentirían sin que nada falle.
    """
    order = list(names.values()) if isinstance(names, dict) else list(names)
    if order != Config.CLASS_NAMES:
        raise RuntimeError(
            f"ERROR  Orden de clases del modelo {order} != Config.CLASS_NAMES {Config.CLASS_NAMES}")


def _is_full_ppe(state: Dict[str, str]) -> bool:
    return all(state[kind] == ok for kind, ok in _PPE_FULL.items())


class RealTimeMaskDetection:
    def __init__(self, ppe_model_path: str) -> None:
        self.model_person: YOLO = self._load_model("yolo11n.pt", "modelo base (personas)")
        self.model_ppe: YOLO = self._load_model(ppe_model_path, "modelo EPI (7 clases)")
        _assert_class_contract(self.model_ppe.names)

        self._init_audio()
        self.last_audio_time: Dict[str, float] = {}
        self.audio_cooldown: float = Config.AUDIO_COOLDOWN

        self.tracked_persons: Dict[int, Dict[str, Any]] = {}
        self._frame_count: int = 0  # Fotogramas vistos; base del expirado de tracks

        self.stats: Dict[str, int] = {
            'total_persons': 0,
            'with_mask': 0,
            'without_mask': 0,
            'incorrect_mask': 0,
            'gloves': 0,
            'no_gloves': 0,
            'cap': 0,
            'no_cap': 0,
            'full_ppe': 0
        }

    def _load_model(self, path: str, description: str) -> YOLO:
        try:
            print(f"Cargando {description}...")
            verify_weight(path)
            model = YOLO(path)
            print(f"OK  {description} cargado correctamente")
            return model
        except FileNotFoundError:
            raise RuntimeError(f"ERROR  Archivo no encontrado: {path}")
        except Exception as e:
            raise RuntimeError(f"ERROR  Cargando {description} ({path}): {e}")

    def _init_audio(self) -> None:
        try:
            pygame.mixer.init()
            self.audio_available = True
        except Exception:
            print("AVISO  Audio no disponible; las alertas serán solo visuales.")
            self.audio_available = False

    def _dec_stat(self, key: str) -> None:
        self.stats[key] = max(0, self.stats[key] - 1)

    def _apply_state(self, state: Dict[str, Any], kind: str, value: str) -> None:
        old = state[kind]
        if old == value:
            return
        if old is not None:
            self._dec_stat(_STAT_KEY[old])
        self.stats[_STAT_KEY[value]] += 1
        if value in _ALERT_ON:
            self.play_audio_alert(value)
        state[kind] = value

    def _update_person_state(self, pid: int, pdata: Dict[str, Any]) -> None:
        """Sin deteccion de un elemento se asume el estado negativo: es la
        direccion en la que un falso negativo no maquilla a un infractor.
        """
        for kind, negative in _DEFAULT_NEGATIVE.items():
            if pdata[kind] is None:
                pdata[kind] = negative

        state = self.tracked_persons.get(pid)
        was_full = False
        if state is None:
            state = {'mask': None, 'gloves': None, 'cap': None}
            self.tracked_persons[pid] = state
            self.stats['total_persons'] += 1
        else:
            # was_full se mide ANTES de aplicar el estado nuevo: si se midiera
            # despues coincidiria siempre con is_full y full_ppe no se moveria.
            was_full = _is_full_ppe(state)

        for kind in _PPE_FULL:
            self._apply_state(state, kind, pdata[kind])

        is_full = _is_full_ppe(state)
        if is_full != was_full:
            if is_full:
                self.stats['full_ppe'] += 1
            else:
                self._dec_stat('full_ppe')

        state['last_seen'] = self._frame_count

    def _cleanup_old_tracks(self) -> None:
        expired = [pid for pid, data in self.tracked_persons.items()
                   if self._frame_count - data['last_seen'] > Config.TRACK_MAX_AGE]
        for pid in expired:
            state = self.tracked_persons.pop(pid)
            for kind in _PPE_FULL:
                self._dec_stat(_STAT_KEY[state[kind]])
            if _is_full_ppe(state):
                self._dec_stat('full_ppe')

    def _throttle_fps(self, loop_start: float) -> None:
        target_fps = Config.TARGET_FPS
        min_frame_time = 1.0 / target_fps
        elapsed = time.monotonic() - loop_start
        sleep_time = min_frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def _play_beep(self) -> None:
        """Nunca propaga excepcion: el audio es opcional."""
        if not self.audio_available:
            return
        try:
            sample_rate = 44100
            duration = 0.3
            freq = 1000
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            tone = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            pygame.mixer.Sound(buffer=tone.tobytes()).play()
        except Exception:
            # Un beep fallido no debe tumbar el bucle de deteccion.
            self.audio_available = False

    def play_audio_alert(self, alert_key: str) -> None:
        if not self.audio_available:
            return
        current_time = time.time()
        last = self.last_audio_time.get(alert_key, 0)
        if current_time - last > self.audio_cooldown:
            mensaje = Config.ALERT_MESSAGES.get(alert_key, alert_key)
            print(f"ALERTA  {mensaje}")

            self._play_beep()

            self.last_audio_time[alert_key] = current_time

    def draw_bbox(self, frame: np.ndarray, box: Any, class_name: str, conf: float, track_id: Optional[int] = None) -> None:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = Config.CLASS_COLORS.get(class_name, (128, 128, 128))
        label = f"{Config.CLASS_DISPLAY_NAMES.get(class_name, class_name)} {conf:.2f}"
        if track_id is not None:
            label = f"ID:{track_id} {label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_person_status(self, frame: np.ndarray, person_box: List[float], track_id: int,
                           gloves_state: str, cap_state: str) -> None:
        x1, y1, x2, y2 = map(int, person_box)

        cv2.putText(frame, f"ID:{track_id}", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset = y2 + 55
        for state in (gloves_state, cap_state):
            if state in _ALERT_ON:
                cv2.putText(frame, Config.CLASS_DISPLAY_NAMES[state], (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.CLASS_COLORS[state], 2)
                y_offset += 25

    def show_statistics(self, frame: np.ndarray) -> None:
        # "unicos" es por id de track, no por persona fisica: tras una oclusion
        # ByteTrack re-adquiere con id nuevo y la misma persona se cuenta dos veces.
        y = 30
        cv2.putText(frame, f"Tracks unicos: {self.stats['total_persons']}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 35

        stats_list: List[Tuple[str, Tuple[int, int, int]]] = [
            (f"Con mascarilla: {self.stats['with_mask']}", Config.CLASS_COLORS['with_mask']),
            (f"Sin mascarilla: {self.stats['without_mask']}", Config.CLASS_COLORS['without_mask']),
            (f"Mal puesta: {self.stats['incorrect_mask']}", Config.CLASS_COLORS['mask_weared_incorrect']),
            (f"Con guantes: {self.stats['gloves']}", Config.CLASS_COLORS['gloves']),
            (f"Sin guantes: {self.stats['no_gloves']}", Config.CLASS_COLORS['no_gloves']),
            (f"Con cofia: {self.stats['cap']}", Config.CLASS_COLORS['cap']),
            (f"Sin cofia: {self.stats['no_cap']}", Config.CLASS_COLORS['no_cap']),
            (f"EPI completo: {self.stats['full_ppe']}", (0, 255, 255))
        ]

        for text, color in stats_list:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

    def run_detection(self) -> None:
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)

        if not cap.isOpened():
            print("ERROR  No se puede acceder a la cámara")
            return

        print("\nOK  Cámara iniciada. Modelos cargados:")
        print("   - Personas: YOLO11n base (tracking activado)")
        print("   - EPI: modelo fine-tuneado con 7 clases")
        print("   Los contadores reflejan el estado ACTUAL de cada persona.")
        print("   ALERTAS ACTIVADAS para: sin mascarilla, mascarilla mal puesta, sin guantes, sin cofia.")
        print("   Presiona 'q' para salir.\n")

        failed_reads = 0

        while True:
            loop_start = time.monotonic()
            self._frame_count += 1

            ret, frame = cap.read()
            if not ret:
                # Una perdida puntual de USB no debe tumbar una sesion de horas:
                # se corta solo tras varios fallos seguidos.
                failed_reads += 1
                if failed_reads >= 10:
                    print("ERROR  La cámara dejó de entregar fotogramas; saliendo.")
                    break
                continue
            failed_reads = 0
            frame = cv2.flip(frame, 1)

            person_results = self.model_person.track(frame, persist=True, verbose=False, classes=[0])[0]
            persons = {}  # track_id -> {'box': [...], 'mask': None, 'gloves': None, 'cap': None}

            if person_results.boxes is not None and person_results.boxes.id is not None:
                for box, tid in zip(person_results.boxes, person_results.boxes.id.cpu().tolist()):
                    conf = float(box.conf[0])
                    if conf < Config.PERSON_CONF_THRESH:
                        continue
                    persons[tid] = {
                        'box': box.xyxy[0].tolist(),
                        'mask': None,
                        'gloves': None,
                        'cap': None
                    }
                    self.draw_bbox(frame, box, 'person', conf, tid)

            ppe_results = self.model_ppe(frame, verbose=False, conf=Config.PPE_CONF_THRESH)[0]

            if ppe_results.boxes is not None:
                for box in ppe_results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model_ppe.names[cls_id]

                    self.draw_bbox(frame, box, class_name, conf, None)

                    bbox = box.xyxy[0].tolist()
                    best_dist = float('inf')
                    best_pid = None
                    for pid, pdata in persons.items():
                        dist = self._distance(bbox, pdata['box'])
                        if dist < best_dist and dist < Config.ASSOCIATION_DIST_THRESH:
                            best_dist = dist
                            best_pid = pid

                    if best_pid is not None:
                        if class_name in ['with_mask', 'without_mask', 'mask_weared_incorrect']:
                            persons[best_pid]['mask'] = class_name
                        elif class_name in ['gloves', 'no_gloves']:
                            persons[best_pid]['gloves'] = class_name
                        elif class_name in ['cap', 'no_cap']:
                            persons[best_pid]['cap'] = class_name

            for pid, pdata in persons.items():
                self._update_person_state(pid, pdata)
                self.draw_person_status(frame, pdata['box'], pid,
                                        pdata['gloves'], pdata['cap'])

            self.show_statistics(frame)
            cv2.putText(frame, "YOLO11n - 7 clases", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            self._cleanup_old_tracks()

            cv2.imshow('Deteccion de EPP - Alertas independientes', frame)

            self._throttle_fps(loop_start)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
