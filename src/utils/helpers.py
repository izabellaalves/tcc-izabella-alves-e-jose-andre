"""Funções utilitárias do pipeline."""

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple


def parse_defects4j_list(content: str) -> List[str]:
    """Parseia listas de linhas retornadas pelo Defects4J."""
    lines = content.strip().split("\n")
    return [line.strip() for line in lines if line.strip()]


def safe_path(path: str) -> Path:
    """Converte string para Path absoluto."""
    return Path(path).resolve()


def count_lines(file_path: Path) -> int:
    """Conta linhas de um arquivo."""
    if not file_path.exists():
        return 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def extract_package_name(full_class_name: str) -> str:
    """Extrai o pacote de um nome completo de classe Java."""
    parts = full_class_name.split(".")
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])


def format_duration(seconds: float) -> str:
    """Formata duração em segundos."""
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = seconds // 60
    secs = seconds % 60

    if minutes < 60:
        return f"{minutes:.0f}min {secs:.0f}s"

    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:.0f}h {mins:.0f}min {secs:.0f}s"


class Timer:
    """Context manager para medir tempo de execução."""

    def __init__(self):
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        return False


def validate_java_class_name(name: str) -> bool:
    """Verifica se a string é um nome válido de classe Java."""
    pattern = r"^[a-zA-Z_$][a-zA-Z0-9_$]*(\.[a-zA-Z_$][a-zA-Z0-9_$]*)*$"
    return bool(re.match(pattern, name))


def parse_test_method_signature(signature: str) -> Tuple[str, str]:
    """Parseia assinatura no formato ClassName::methodName."""
    if "::" in signature:
        parts = signature.split("::")
    elif "." in signature and signature.count(".") >= 1:
        parts = signature.rsplit(".", 1)
    else:
        raise ValueError(f"Formato inválido de assinatura: {signature}")

    if len(parts) != 2:
        raise ValueError(f"Formato inválido de assinatura: {signature}")

    return parts[0].strip(), parts[1].strip()


def ensure_dir(path: Path) -> Path:
    """Cria diretório se não existir."""
    path.mkdir(parents=True, exist_ok=True)
    return path
