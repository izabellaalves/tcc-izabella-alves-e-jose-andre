"""Configurações do ambiente e execução."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .constants import (
    DATA_DIR,
    DEFECTS4J_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    PROJECTS,
    REQUIRED_JAVA_VERSION,
)


@dataclass
class Settings:
    """Configurações centralizadas do projeto."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    logs_dir: Path = LOGS_DIR
    defects4j_dir: Path = DEFECTS4J_DIR
    log_level: int = logging.INFO
    log_to_console: bool = True
    log_to_file: bool = True
    required_java_version: int = REQUIRED_JAVA_VERSION
    skip_checkout: bool = False
    projects_to_process: List[str] = None

    def __post_init__(self):
        if self.projects_to_process is None:
            self.projects_to_process = PROJECTS.copy()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            log_level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
            skip_checkout=os.getenv("SKIP_CHECKOUT", "").lower() == "true",
        )
