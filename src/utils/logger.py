"""Configuração de logging do pipeline."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "tcc_pipeline",
    log_dir: Path = Path("logs"),
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """Configura logging para arquivo e console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("Logger inicializado. Log salvo em: %s", log_file)

    return logger


def get_logger(name: str = "tcc_pipeline") -> logging.Logger:
    """Retorna o logger configurado."""
    return logging.getLogger(name)


class BugProcessingLogger:
    """Context manager para logging de processamento de um bug."""

    def __init__(self, logger: logging.Logger, project: str, bug_id: int):
        self.logger = logger
        self.project = project
        self.bug_id = bug_id
        self.bug_name = f"{project}-{bug_id}"
        self.start_time: Optional[datetime] = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info("Iniciando processamento: %s", self.bug_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info("%s concluído em %.2fs", self.bug_name, elapsed)
        else:
            self.logger.error(
                "%s falhou após %.2fs: %s: %s",
                self.bug_name,
                elapsed,
                exc_type.__name__,
                exc_val,
            )

        return False

    def info(self, message: str, *args, **kwargs):
        self.logger.info(f"[{self.bug_name}] {message}", *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(f"[{self.bug_name}] {message}", *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self.logger.error(f"[{self.bug_name}] {message}", *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(f"[{self.bug_name}] {message}", *args, **kwargs)
