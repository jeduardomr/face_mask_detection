import hashlib
from pathlib import Path


def _manifest() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "weights.sha256"
        if candidate.is_file():
            return candidate
    return None


def _entries() -> dict:
    manifest = _manifest()
    if manifest is None:
        return {}
    table = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if len(digest) == 64 and name.strip():
            table[name.strip()] = digest.strip()
    return table


def verify_weight(path: str) -> None:
    """Aborta si el .pt no coincide con el hash registrado en weights.sha256.

    Sin hash conocido no hay contra qué comparar, así que esos pesos se aceptan.
    """
    key = Path(path).as_posix()
    want = _entries().get(key) or _entries().get(Path(key).name)
    if want is None:
        return
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != want:
        # El caso normal de este fallo es haber reentrenado sin refrescar el
        # manifiesto; sin esta pista el mensaje parece un ataque o corrupcion.
        raise RuntimeError(
            f"ERROR  Peso modificado o corrupto: {path}\n"
            f"   Si acabas de reentrenar, regenera la entrada: sha256sum {key} >> weights.sha256")
