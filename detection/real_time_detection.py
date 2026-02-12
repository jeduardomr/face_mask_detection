import cv2
from ultralytics import YOLO
import pygame
import time
import winsound
from utils.config import Config

class RealTimeMaskDetection:
    def __init__(self, ppe_model_path):
        """
        Carga dos modelos:
        - model_person: YOLO11n base (solo clase 'person')
        - model_ppe: modelo fine-tuneado con 7 clases
        """
        print("🔄 Cargando modelo base para detección de personas...")
        self.model_person = YOLO("yolo11n.pt")
        print("🔄 Cargando modelo de EPI (7 clases)...")
        self.model_ppe = YOLO(ppe_model_path)

        pygame.mixer.init()
        # Diccionario para cooldown por tipo de alerta
        self.last_audio_time = {}
        self.audio_cooldown = 5  # segundos

        # Almacén de personas trackeadas con su último estado
        self.tracked_persons = {}

        # Estadísticas acumuladas (SIEMPRE reflejan el estado actual)
        self.stats = {
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

    def _distance(self, bbox1, bbox2):
        """Distancia euclidiana entre centros de dos bounding boxes"""
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def play_audio_alert(self, alert_key):
        """
        Reproduce alerta de audio con cooldown INDEPENDIENTE por cada tipo.
        En Windows usa winsound.Beep; puedes cambiarlo por pygame.mixer.Sound.
        """
        current_time = time.time()
        last = self.last_audio_time.get(alert_key, 0)
        if current_time - last > self.audio_cooldown:
            mensaje = Config.ALERT_MESSAGES.get(alert_key, alert_key)
            print(f"🔊 ALERTA: {mensaje}")
            
            # --- REPRODUCIR SONIDO REAL ---
            winsound.Beep(1000, 300)  # 1000 Hz, 300 ms

            self.last_audio_time[alert_key] = current_time

    def draw_bbox(self, frame, box, class_name, conf, track_id=None):
        """Dibuja bounding box con etiqueta y color desde Config"""
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = Config.CLASS_COLORS.get(class_name, (128, 128, 128))
        display_name = Config.CLASS_DISPLAY_NAMES.get(class_name, class_name)  # ← TRADUCCIÓN
        label = f"{display_name} {conf:.2f}"

        if track_id is not None:
            label = f"ID:{track_id} {display_name} {conf:.2f}"
        else:
            label = f"{display_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_person_status(self, frame, person_box, track_id, mask_state, gloves_state, cap_state):
        """Dibuja iconos y textos de ausencia debajo de la persona"""
        x1, y1, x2, y2 = map(int, person_box)

        # Línea 1: iconos de estado
        mask_icon = "😷" if mask_state == 'with_mask' else "😨" if mask_state == 'without_mask' else "😕"
        gloves_icon = "🧤" if gloves_state == 'gloves' else "✋"
        cap_icon = "⛑️" if cap_state == 'cap' else "👤"
        status_line = f"ID:{track_id} {mask_icon} {gloves_icon} {cap_icon}"
        cv2.putText(frame, status_line, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Línea 2: textos de ausencia (refuerzo visual)
        y_offset = y2 + 55
        if gloves_state == 'no_gloves':
            text_no = Config.CLASS_DISPLAY_NAMES.get('no_gloves', 'Sin guantes')
            color_no = Config.CLASS_COLORS.get('no_gloves', (128, 0, 128))
            cv2.putText(frame, f"❌ {text_no}", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_no, 2)
            y_offset += 25
        if cap_state == 'no_cap':
            text_no = Config.CLASS_DISPLAY_NAMES.get('no_cap', 'Sin cofia')
            color_no = Config.CLASS_COLORS.get('no_cap', (0, 255, 255))
            cv2.putText(frame, f"❌ {text_no}", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_no, 2)

    def show_statistics(self, frame):
        """Muestra estadísticas acumuladas (estado actual de todas las personas)"""
        y = 30
        cv2.putText(frame, f"Personas unicas: {self.stats['total_persons']}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 35

        stats_list = [
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

    def run_detection(self):
        """Bucle principal con dos modelos y actualización dinámica de contadores"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)

        if not cap.isOpened():
            print("❌ Error: No se puede acceder a la cámara")
            return

        print("\n✅ Cámara iniciada. Modelos cargados:")
        print("   - Personas: YOLO11n base (tracking activado)")
        print("   - EPI: modelo fine-tuneado con 7 clases")
        print("   📊 Los contadores reflejan el estado ACTUAL de cada persona.")
        print("   🔊 ALERTAS ACTIVADAS para: sin mascarilla, mascarilla mal puesta, sin guantes, sin cofia.")
        print("   Presiona 'q' para salir.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # ---------- 1. DETECTAR PERSONAS (modelo base, solo clase 0) ----------
            person_results = self.model_person.track(frame, persist=True, verbose=False, classes=[0])[0]
            persons = {}  # track_id -> {'box': [...], 'mask': None, 'gloves': None, 'cap': None}

            if person_results.boxes is not None and person_results.boxes.id is not None:
                for box, tid in zip(person_results.boxes, person_results.boxes.id.cpu().tolist()):
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue
                    persons[tid] = {
                        'box': box.xyxy[0].tolist(),
                        'mask': None,
                        'gloves': None,
                        'cap': None
                    }
                    self.draw_bbox(frame, box, 'person', conf, tid)

            # ---------- 2. DETECTAR EPI (modelo fine-tuneado) ----------
            ppe_results = self.model_ppe(frame, verbose=False, conf=0.3)[0]

            if ppe_results.boxes is not None:
                for box in ppe_results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model_ppe.names[cls_id]

                    # Dibujar siempre el EPI detectado
                    self.draw_bbox(frame, box, class_name, conf, None)

                    # Asociar a la persona más cercana
                    bbox = box.xyxy[0].tolist()
                    best_dist = float('inf')
                    best_pid = None
                    for pid, pdata in persons.items():
                        dist = self._distance(bbox, pdata['box'])
                        if dist < best_dist and dist < 200:  # umbral de distancia
                            best_dist = dist
                            best_pid = pid

                    if best_pid is not None:
                        if class_name in ['with_mask', 'without_mask', 'mask_weared_incorrect']:
                            persons[best_pid]['mask'] = class_name
                        elif class_name in ['gloves', 'no_gloves']:
                            persons[best_pid]['gloves'] = class_name
                        elif class_name in ['cap', 'no_cap']:
                            persons[best_pid]['cap'] = class_name

            # ---------- 3. PROCESAR CADA PERSONA: ACTUALIZAR ESTADÍSTICAS DINÁMICAMENTE ----------
            for pid, pdata in persons.items():
                # --- APLICAR INFERENCIA: si no hay detección, asumimos estado por defecto ---
                current_mask = pdata['mask']
                if current_mask is None:
                    current_mask = 'without_mask'
                    pdata['mask'] = 'without_mask'

                current_gloves = pdata['gloves']
                if current_gloves is None:
                    current_gloves = 'no_gloves'
                    pdata['gloves'] = 'no_gloves'

                current_cap = pdata['cap']
                if current_cap is None:
                    current_cap = 'no_cap'
                    pdata['cap'] = 'no_cap'

                # --- NUEVA PERSONA ---
                if pid not in self.tracked_persons:
                    self.tracked_persons[pid] = {
                        'mask': current_mask,
                        'gloves': current_gloves,
                        'cap': current_cap
                    }
                    self.stats['total_persons'] += 1

                    # Mascarilla inicial
                    if current_mask == 'with_mask':
                        self.stats['with_mask'] += 1
                    elif current_mask == 'without_mask':
                        self.stats['without_mask'] += 1
                        self.play_audio_alert('without_mask')
                    elif current_mask == 'mask_weared_incorrect':
                        self.stats['incorrect_mask'] += 1
                        self.play_audio_alert('mask_weared_incorrect')

                    # Guantes iniciales
                    if current_gloves == 'gloves':
                        self.stats['gloves'] += 1
                    else:  # no_gloves
                        self.stats['no_gloves'] += 1
                        self.play_audio_alert('no_gloves')

                    # Cofia inicial
                    if current_cap == 'cap':
                        self.stats['cap'] += 1
                    else:  # no_cap
                        self.stats['no_cap'] += 1
                        self.play_audio_alert('no_cap')

                    # EPI completo inicial
                    if current_mask == 'with_mask' and current_gloves == 'gloves' and current_cap == 'cap':
                        self.stats['full_ppe'] += 1

                # --- PERSONA YA EXISTENTE: VERIFICAR CAMBIOS DE ESTADO ---
                else:
                    prev = self.tracked_persons[pid]

                    # ---- CAMBIO DE MASCARILLA ----
                    if prev['mask'] != current_mask:
                        # Decrementar estado anterior
                        if prev['mask'] == 'with_mask':
                            self.stats['with_mask'] -= 1
                        elif prev['mask'] == 'without_mask':
                            self.stats['without_mask'] -= 1
                        elif prev['mask'] == 'mask_weared_incorrect':
                            self.stats['incorrect_mask'] -= 1

                        # Incrementar nuevo estado
                        if current_mask == 'with_mask':
                            self.stats['with_mask'] += 1
                        elif current_mask == 'without_mask':
                            self.stats['without_mask'] += 1
                            self.play_audio_alert('without_mask')
                        elif current_mask == 'mask_weared_incorrect':
                            self.stats['incorrect_mask'] += 1
                            self.play_audio_alert('mask_weared_incorrect')

                        prev['mask'] = current_mask

                    # ---- CAMBIO DE GUANTES ----
                    if prev['gloves'] != current_gloves:
                        if prev['gloves'] == 'gloves':
                            self.stats['gloves'] -= 1
                        else:  # no_gloves
                            self.stats['no_gloves'] -= 1

                        if current_gloves == 'gloves':
                            self.stats['gloves'] += 1
                        else:  # no_gloves
                            self.stats['no_gloves'] += 1
                            self.play_audio_alert('no_gloves')

                        prev['gloves'] = current_gloves

                    # ---- CAMBIO DE COFIA ----
                    if prev['cap'] != current_cap:
                        if prev['cap'] == 'cap':
                            self.stats['cap'] -= 1
                        else:  # no_cap
                            self.stats['no_cap'] -= 1

                        if current_cap == 'cap':
                            self.stats['cap'] += 1
                        else:  # no_cap
                            self.stats['no_cap'] += 1
                            self.play_audio_alert('no_cap')

                        prev['cap'] = current_cap

                    # ---- ACTUALIZAR EPI COMPLETO ----
                    was_full = (prev['mask'] == 'with_mask' and prev['gloves'] == 'gloves' and prev['cap'] == 'cap')
                    is_full  = (current_mask == 'with_mask' and current_gloves == 'gloves' and current_cap == 'cap')
                    if was_full != is_full:
                        if is_full:
                            self.stats['full_ppe'] += 1
                        else:
                            self.stats['full_ppe'] -= 1

                # Dibujar estado visual de la persona (siempre con valores actuales)
                self.draw_person_status(frame, pdata['box'], pid,
                                        current_mask, current_gloves, current_cap)

            # ---------- 4. MOSTRAR ESTADÍSTICAS ----------
            self.show_statistics(frame)
            cv2.putText(frame, "YOLO11n - 7 clases", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Deteccion de EPP - Alertas independientes', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()