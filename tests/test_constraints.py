# python3 tests/test_constraints.py  (bloqueante, <2s, sin deps)
"""Comprueba el floor de calidad del proyecto sobre el arbol de trabajo."""
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"data", "runs", "models", "venv", "__pycache__", "node_modules"}
BUILD_BUDGET_MB = 100
failures: list[str] = []


def fail(rule: str, detail: str) -> None:
    failures.append(f"{rule}: {detail}")


def py_files() -> list[Path]:
    # startswith(".") cubre ademas los directorios ocultos, incluido el de 61 MB.
    return [p for p in ROOT.rglob("*.py")
            if not any(part in SKIP or part.startswith(".") for part in p.relative_to(ROOT).parts)]


# Los checks de comentarios no son solo para .py: un Dockerfile o un script se
# comentan igual de facil, y dejarlos fuera hace decorativa la regla.
CONF_FILES = [ROOT / ".gitignore", ROOT / ".dockerignore", ROOT / "Dockerfile",
              ROOT / "docker-compose.yml", ROOT / "xhost_setup.sh",
              ROOT / ".devcontainer" / "devcontainer.json",
              ROOT / ".devcontainer" / "docker-compose.devcontainer.yml",
              ROOT / ".devcontainer" / "post-create.sh"]


# --- Floor: sin supresiones ---------------------------------------------
SUPPRESSIONS = ("noqa", "type: ignore", "nosemgrep", "gitleaks:allow", "pylint: disable")
for path in py_files():
    rel = path.relative_to(ROOT)
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if rel.parts[0] == "tests" or path.name == "test_constraints.py":
            continue  # el detector no se escanea a si mismo
        if any(marker in line for marker in SUPPRESSIONS):
            fail("sin supresiones", f"{rel}:{n}")

# --- Floor: sin except/catch vacio que trague el fallo -------------------
for path in py_files():
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and len(node.body) == 1:
            if isinstance(node.body[0], (ast.Pass, ast.Continue)):
                fail("sin except vacio", f"{path.relative_to(ROOT)}:{node.lineno}")


# --- Floor: comentarios que calcan el identificador --------------------
# Alcance DELIBERADAMENTE estrecho: solo pilla el comentario que repite el
# identificador literal de la linea siguiente ("Return the total" sobre
# `return total`). No intenta juzgar si un comentario en castellano esta de
# sobra ("Crear el directorio" sobre os.makedirs) — eso no lo decide un
# patron, lo decide quien lee el comentario.
SIGNIFICANT_WORDS = {
    "a", "al", "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "y", "o", "en", "con", "por", "para", "se", "su", "sus", "lo", "le", "que",
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "on", "is", "it",
}
MIN_TOKEN_LEN = 4


def comment_body(text: str) -> str | None:
    """Cuerpo del comentario, o None si la línea no es un comentario.

    `//` es para los .jsonc (devcontainer.json); el resto del repo comenta con `#`.
    """
    for marker in ("#", "//"):
        if text.startswith(marker):
            return text[len(marker):].strip()
    return None


def comment_echoes_next_line(path: Path) -> list[tuple[int, str]]:
    """Comentarios que repiten el identificador de la línea siguiente."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for i, raw in enumerate(lines[:-1]):
        text = raw.strip()
        comment = comment_body(text)
        if comment is None:
            continue
        # Secciones demarcadas y shebangs no cuentan.
        if not comment or comment.startswith(("=", "-", "!", "/")) or set(comment) <= {"-", "="}:
            continue

        nxt = lines[i + 1].strip()
        if not nxt or comment_body(nxt) is not None:
            continue

        words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", comment.lower())
                 if w not in SIGNIFICANT_WORDS and len(w) >= MIN_TOKEN_LEN]
        if not words:
            continue

        # Identificadores y palabras clave de la linea siguiente. Se descartan
        # los literales: un string no es el "nombre" que el codigo ya declara.
        code = re.sub(r'"[^"]*"|\'[^\']*\'', " ", nxt)
        code_ids = [w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
                    if w not in ("self", "cls")]
        if not code_ids:
            continue
        matches = sum(
            1 for w in words
            if any(w == c or c.startswith(w) or w.startswith(c) and len(c) >= MIN_TOKEN_LEN
                   for c in code_ids)
        )
        if matches / len(words) >= 0.5:
            out.append((i + 1, comment))
    return out


comment_files = [p for p in py_files()
                 if p.relative_to(ROOT).parts[0] != "tests"
                 and p.name != "test_constraints.py"] + CONF_FILES

for path in comment_files:
    rel = path.relative_to(ROOT)
    for lineno, text in comment_echoes_next_line(path):
        fail("comentario que calca el codigo",
             f"{rel}:{lineno}  «{text[:60]}»")

# --- Floor: comentarios que solo narran lo evidente --------------------
# Los comentarios que empiezan con "crear/asignar/iniciar/return/cargar..."
# y NO explican un "por que" se leen como ruido. Se allowlistan los que
# aportan contexto (contienen un porque, un motivo o una condicion).
VERB_ONLY = re.compile(
    r"^\s*(#\s*)?(crea\w*|asigna\w*|inicia\w*|devuelve\w*|retorna\w*|carga\w*|"
    r"guarda\w*|añade\w*|agrega\w*|copia\w*|muestra\w*|imprime\w*|llama\w*|"
    r"instala\w*|abre\w*|cierra\w*|recorre\w*|busca\w*|filtra\w*|ordena\w*|"
    r"creat\w*|set\w*|return\w*|add\w*|copy\w*|show\w*|print\w*|call\w*|"
    r"open\w*|close\w*|loop\w*|iterate\w*)\b",
    re.IGNORECASE)
MOTIVE = re.compile(r"\b(para|porque|por qué|si |cuando|evita|no |nunca|siempre|"
                    r"sin |antes|después|tiene que|debe|solo si|aunque|"
                    r"because|so that|instead|rather|avoid|never|always|"
                    r"without|unless|when)\b", re.IGNORECASE)

for path in comment_files:
    rel = path.relative_to(ROOT)
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        body = comment_body(text)
        if not body or body.startswith(("=", "-", "!", "/")):
            continue
        if VERB_ONLY.match(body) and not MOTIVE.search(body):
            fail("comentario que solo narra lo evidente",
                 f"{rel}:{n}  «{body[:60]}» — explica el porqué o bórralo")

# --- Floor: emojis en comentarios (no en strings) ----------------------
EMOJI_RANGES = ((0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F000, 0x1F02F), (0x2139, 0x2139))
for path in comment_files:
    rel = path.relative_to(ROOT)
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        # Sobre la linea entera y no solo el comentario: cv2.putText pinta '?'
        # en vez de emoji, y la salida va con etiquetas ASCII.
        if any(lo <= ord(ch) <= hi for ch in line for lo, hi in EMOJI_RANGES):
            fail("sin emojis", f"{rel}:{n}")

# --- Floor: sin codigo muerto en Config --------------------------------
CONFIG = ROOT / "utils" / "config.py"
tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config")
declared = set()
for stmt in cls.body:
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        declared.add(stmt.target.id)
    elif isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
        declared.add(stmt.targets[0].id)

users = {p: p.read_text(encoding="utf-8") for p in py_files()}
for const in sorted(declared):
    if const.startswith("_"):
        continue
    reads = sum(t.count(f"Config.{const}") for t in users.values())
    reads += sum(t.count(f"import Config") and t.count(const) for t in users.values()
                 if "import Config" in t)
    if reads == 0:
        fail("sin constantes muertas", f"Config.{const} no la lee nadie")

# --- Enforced: todo compila --------------------------------------------
compile_result = subprocess.run(
    [sys.executable, "-m", "compileall", "-q",
     *[str(p) for p in py_files()]],
    capture_output=True, text=True)
if compile_result.returncode != 0:
    fail("compila", compile_result.stdout.strip() or compile_result.stderr.strip()[:200])

# --- Enforced: .dockerignore presente y con las exclusiones criticas ----
dockerignore = ROOT / ".dockerignore"
if not dockerignore.exists():
    fail("dockerignore", "no existe .dockerignore; COPY . . arrastrara todo")
else:
    patterns = {ln.strip() for ln in dockerignore.read_text().splitlines() if ln.strip()}
    for needed in (".git/", "data/", "models/"):
        if needed not in patterns:
            fail("dockerignore", f"falta la exclusion {needed}")

    required = {"runs/**/*.jpg", "runs/detect/*/weights/epoch*.pt", "runs/detect/*/weights/last.pt"}
    for needed in sorted(required):
        if needed not in patterns:
            fail("dockerignore", f"falta excluir {needed}")

# --- Enforced: contexto de build dentro de presupuesto -----------------
# Se mide lo que el .dockerignore deja pasar, no el arbol entero.
def context_size_mb(patterns: list[str]) -> float:
    """Suma lo que dejaria pasar un .dockerignore con esos patrones."""
    total = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(ROOT)
        if any(rel.match(p.rstrip("/")) or str(rel).startswith(p.rstrip("/") + "/")
               for p in patterns if p.endswith("/")):
            continue
        if any(rel.match(p) for p in patterns if not p.endswith("/")):
            continue
        total += path.stat().st_size
    return total / 1e6


docker_patterns = [ln.strip() for ln in dockerignore.read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")]
build_mb = context_size_mb(docker_patterns)
if build_mb > BUILD_BUDGET_MB:
    fail("contexto de build",
         f"{build_mb:.0f} MB superan el presupuesto de {BUILD_BUDGET_MB} MB")

if not failures:
    print(f"CONSTRAINTS: OK  (contexto de build {build_mb:.0f} MB / {BUILD_BUDGET_MB} MB)")
    sys.exit(0)

print("CONSTRAINTS: BLOQUEADO\n")
for f in failures:
    print(f"  - {f}")
sys.exit(1)