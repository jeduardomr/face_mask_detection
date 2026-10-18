# python3 tests/test_state_machine.py  (sin pytest, sin GPU, sin camara)
"""Ejercita la maquina de estados de RealTimeMaskDetection sin camara ni torch.

Stubea cv2 / ultralytics / pygame y llama a los metodos reales de la clase
(instanciada con __new__, sin pasar por __init__ y sin cargar pesos).
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for name in ("cv2", "ultralytics", "pygame"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["ultralytics"].YOLO = object
sys.modules["pygame"].mixer = types.SimpleNamespace(init=lambda: None)

from utils.config import Config  # noqa: E402
from detection.real_time_detection import RealTimeMaskDetection, _assert_class_contract  # noqa: E402


def make_detector() -> RealTimeMaskDetection:
    d = RealTimeMaskDetection.__new__(RealTimeMaskDetection)
    d._frame_count = 0
    d.tracked_persons = {}
    d.last_audio_time = {}
    d.audio_cooldown = Config.AUDIO_COOLDOWN
    d.audio_available = False
    d.stats = dict.fromkeys(
        ["total_persons", "with_mask", "without_mask", "incorrect_mask",
         "gloves", "no_gloves", "cap", "no_cap", "full_ppe"], 0)
    return d


def frame(det: RealTimeMaskDetection, pid: int, mask, gloves, cap) -> None:
    """Un fotograma del bucle, sin dibujo: llama al metodo real de produccion."""
    det._frame_count += 1
    pdata = {"box": [0, 0, 10, 10], "mask": mask, "gloves": gloves, "cap": cap}
    det._update_person_state(pid, pdata)


failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got}" + ("" if ok else f"  != esperado {want}"))
    if not ok:
        failures.append(name)


# 1. Persona nueva completa -> full_ppe sube
d = make_detector()
frame(d, 1, "with_mask", "gloves", "cap")
check("nueva completa full_ppe", d.stats["full_ppe"], 1)
check("nueva completa total_persons", d.stats["total_persons"], 1)
check("nueva completa with_mask", d.stats["with_mask"], 1)

# 2. Se quita los guantes -> full_ppe DEBE bajar (este es el bug que estaba congelado)
frame(d, 1, "with_mask", "no_gloves", "cap")
check("quita guantes -> full_ppe baja", d.stats["full_ppe"], 0)
check("quita guantes -> no_gloves", d.stats["no_gloves"], 1)
check("quita guantes -> gloves baja", d.stats["gloves"], 0)

# 3. Se los vuelve a poner -> full_ppe vuelve a subir
frame(d, 1, "with_mask", "gloves", "cap")
check("repone guantes -> full_ppe sube", d.stats["full_ppe"], 1)
check("repone guantes -> gloves", d.stats["gloves"], 1)

# 4. Mascarilla incorrecta cuenta en 'incorrect_mask', no en 'without_mask'
d2 = make_detector()
frame(d2, 7, "mask_weared_incorrect", None, None)
check("mascarilla mal puesta", d2.stats["incorrect_mask"], 1)
check("mascarilla mal puesta no suma without_mask", d2.stats["without_mask"], 0)
check("sin deteccion -> no_gloves", d2.stats["no_gloves"], 1)
check("sin deteccion -> no_cap", d2.stats["no_cap"], 1)

# 5. Expiracion del track descuenta lo que sumo
d3 = make_detector()
frame(d3, 2, "with_mask", "gloves", "cap")
for _ in range(Config.TRACK_MAX_AGE + 2):
    d3._frame_count += 1
d3._cleanup_old_tracks()
check("track expirado -> tracked vacio", len(d3.tracked_persons), 0)
check("track expirado -> full_ppe", d3.stats["full_ppe"], 0)
check("track expirado -> with_mask", d3.stats["with_mask"], 0)
check("track expirado -> gloves", d3.stats["gloves"], 0)
check("track expirado -> cap", d3.stats["cap"], 0)
check("track expirado -> total_persons sigue", d3.stats["total_persons"], 1)

# 6. Invariante: suma de contadores por estado == nº de tracks vivos
d4 = make_detector()
for i, (m, g, c) in enumerate([
    ("with_mask", "gloves", "cap"),
    ("without_mask", "no_gloves", "no_cap"),
    ("mask_weared_incorrect", "gloves", "no_cap"),
    ("with_mask", "gloves", "no_cap"),
]):
    frame(d4, i, m, g, c)
    frame(d4, i, m, g, c)  # repetir no debe mover contadores
check("tracks vivos", len(d4.tracked_persons), 4)
check("invariante with_mask", d4.stats["with_mask"], 2)
check("invariante without_mask", d4.stats["without_mask"], 1)
check("invariante incorrect_mask", d4.stats["incorrect_mask"], 1)
check("invariante gloves", d4.stats["gloves"], 3)
check("invariante no_gloves", d4.stats["no_gloves"], 1)
check("invariante cap", d4.stats["cap"], 1)
check("invariante no_cap", d4.stats["no_cap"], 3)
check("invariante full_ppe", d4.stats["full_ppe"], 1)

def contract_holds(names) -> bool:
    try:
        _assert_class_contract(names)
        return True
    except RuntimeError:
        return False


# 7. Orden de clases: el modelo debe traer la lista de Config, ni una mas ni menos
check("orden correcto aceptado",
      contract_holds(dict(enumerate(Config.CLASS_NAMES))), True)
check("orden alterado rechazado",
      contract_holds(dict(enumerate(reversed(Config.CLASS_NAMES)))), False)
check("clase de mas rechazada",
      contract_holds({**dict(enumerate(Config.CLASS_NAMES)), 7: "extra"}), False)

print()
print("TODO OK" if not failures else f"FALLARON: {failures}")
sys.exit(1 if failures else 0)