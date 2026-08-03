"""Validação do ambiente de execução."""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

from config.constants import DEFECTS4J_DIR, REQUIRED_JAVA_VERSION
from src.utils.logger import get_logger


@dataclass
class EnvironmentConfig:
    """Configuração do ambiente validado."""

    env_type: Literal["docker", "local"]
    java_version: str
    java_home: Path
    python_version: str
    defects4j_path: Path
    defects4j_cmd: str
    perl_available: bool
    git_available: bool


def detect_environment() -> Literal["docker", "local"]:
    """Detecta se a execução ocorre em Docker ou localmente."""
    if Path("/.dockerenv").exists():
        return "docker"

    cgroup_path = Path("/proc/1/cgroup")
    if cgroup_path.exists():
        try:
            content = cgroup_path.read_text(encoding="utf-8")
            if "docker" in content or "containerd" in content:
                return "docker"
        except OSError:
            pass

    return "local"


def get_java_version() -> Optional[Tuple[str, int]]:
    """Retorna versão do Java instalado ou None."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        output = result.stderr + result.stdout

        match = re.search(r'version\s+"?(\d+)\.(\d+)\.(\d+)', output)
        if match:
            major = int(match.group(1))
            full_version = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
            return full_version, major

        match = re.search(r'version\s+"?(\d+)', output)
        if match:
            major = int(match.group(1))
            return str(major), major

        return None

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return None


def check_command_available(command: str) -> bool:
    """Verifica se um comando está disponível no PATH."""
    try:
        subprocess.run(
            [command, "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def find_defects4j_path() -> Optional[Path]:
    """Localiza o diretório do Defects4J."""
    env_type = detect_environment()

    if env_type == "docker":
        docker_path = Path("/defects4j")
        if docker_path.exists():
            return docker_path

    if DEFECTS4J_DIR.exists():
        return DEFECTS4J_DIR

    windows_path = Path("c:/Users/Softex/Desktop/TCC2/defects4j")
    if windows_path.exists():
        return windows_path

    return None


def get_defects4j_command(defects4j_path: Path, env_type: str) -> str:
    """Retorna o comando para executar o Defects4J."""
    if env_type == "docker":
        return "defects4j"

    if check_command_available("defects4j"):
        return "defects4j"

    script_path = defects4j_path / "framework" / "bin" / "defects4j"
    if sys.platform == "win32":
        return f"perl {script_path}"

    return str(script_path)


def validate_environment(
    required_java_version: int = REQUIRED_JAVA_VERSION,
) -> EnvironmentConfig:
    """Valida o ambiente e retorna a configuração."""
    logger = get_logger()

    env_type = detect_environment()
    logger.info("Ambiente detectado: %s", env_type)

    python_version = (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    logger.info("Python: %s", python_version)

    if sys.version_info < (3, 8):
        raise RuntimeError(
            f"Python 3.8+ requerido. Versão atual: {python_version}"
        )

    java_info = get_java_version()
    if java_info is None:
        raise RuntimeError(
            "Java não encontrado. Use Docker ou instale Java 11."
        )

    java_version_str, java_major = java_info
    logger.info("Java: %s (major: %d)", java_version_str, java_major)

    if java_major != required_java_version:
        warning_msg = (
            f"Java {java_major} detectado, mas Defects4J 3.0.1 requer "
            f"Java {required_java_version}. Bugs podem não compilar corretamente."
        )

        if env_type == "local":
            raise RuntimeError(
                f"{warning_msg}\nUse Docker ou instale Java 11."
            )

        logger.warning(warning_msg)

    java_home = Path(os.environ.get("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64"))
    try:
        output = subprocess.run(
            ["java", "-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stderr

        for line in output.split("\n"):
            if "java.home" in line:
                java_home = Path(line.split("=")[1].strip())
                break
    except Exception:
        pass

    logger.info("JAVA_HOME: %s", java_home)

    perl_available = check_command_available("perl")
    logger.info("Perl disponível: %s", perl_available)

    if env_type == "local" and not perl_available:
        raise RuntimeError(
            "Perl não encontrado. Use Docker ou instale Perl manualmente."
        )

    git_available = check_command_available("git")
    logger.info("Git disponível: %s", git_available)

    defects4j_path = find_defects4j_path()
    if defects4j_path is None:
        raise RuntimeError(
            "Defects4J não encontrado. Esperado em ./defects4j/ ou /defects4j/."
        )

    logger.info("Defects4J encontrado em: %s", defects4j_path)

    readme_path = defects4j_path / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if "3.0.1" in first_line:
                logger.info("Defects4J versão: 3.0.1")
            else:
                logger.warning(
                    "Versão do Defects4J não confirmada: %s",
                    first_line.strip(),
                )

    defects4j_cmd = get_defects4j_command(defects4j_path, env_type)
    logger.info("Comando Defects4J: %s", defects4j_cmd)

    config = EnvironmentConfig(
        env_type=env_type,
        java_version=java_version_str,
        java_home=java_home,
        python_version=python_version,
        defects4j_path=defects4j_path,
        defects4j_cmd=defects4j_cmd,
        perl_available=perl_available,
        git_available=git_available,
    )

    logger.info("Ambiente validado com sucesso")

    return config
