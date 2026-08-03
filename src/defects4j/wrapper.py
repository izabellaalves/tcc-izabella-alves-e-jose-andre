"""Wrapper para comandos do Defects4J."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils.environment import EnvironmentConfig
from src.utils.logger import get_logger


@dataclass
class CommandResult:
    """Resultado da execução de um comando."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    elapsed_seconds: float


class Defects4JWrapper:
    """Executa checkout, compile e export do Defects4J."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.logger = get_logger()
        self.defects4j_cmd = config.defects4j_cmd
        self.default_timeout = 300

    def _run_command(
        self,
        args: List[str],
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        timeout = timeout or self.default_timeout
        start_time = time.time()

        try:
            self.logger.debug(
                "Executando: %s (cwd: %s, timeout: %ss)",
                " ".join(args),
                cwd,
                timeout,
            )

            result = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )

            elapsed = time.time() - start_time

            return CommandResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                elapsed_seconds=elapsed,
            )

        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            self.logger.error("Comando expirou após %ss", timeout)

            return CommandResult(
                success=False,
                stdout=e.stdout.decode("utf-8", errors="replace") if e.stdout else "",
                stderr=e.stderr.decode("utf-8", errors="replace") if e.stderr else "",
                exit_code=-1,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error("Erro ao executar comando: %s", e)

            return CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-2,
                elapsed_seconds=elapsed,
            )

    def checkout(
        self,
        project: str,
        bug_id: int,
        version: str,
        work_dir: Path,
    ) -> CommandResult:
        """Faz checkout de um bug."""
        work_dir.parent.mkdir(parents=True, exist_ok=True)

        if self.config.env_type == "local" and "perl" in self.defects4j_cmd:
            args = self.defects4j_cmd.split() + [
                "checkout",
                "-p",
                project,
                "-v",
                f"{bug_id}{version}",
                "-w",
                str(work_dir),
            ]
        else:
            args = [
                self.defects4j_cmd,
                "checkout",
                "-p",
                project,
                "-v",
                f"{bug_id}{version}",
                "-w",
                str(work_dir),
            ]

        self.logger.info("Checkout %s-%s%s -> %s", project, bug_id, version, work_dir)
        result = self._run_command(args, timeout=600)

        # Defects4J pode retornar exit code != 0 mesmo com checkout bem-sucedido
        config_file = work_dir / ".defects4j.config"
        actual_success = config_file.exists()

        if actual_success:
            self.logger.info("Checkout concluído em %.2fs", result.elapsed_seconds)
            return CommandResult(
                success=True,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                elapsed_seconds=result.elapsed_seconds,
            )

        self.logger.error(
            "Checkout falhou (exit %d): %s",
            result.exit_code,
            result.stderr[:200],
        )
        return CommandResult(
            success=False,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            elapsed_seconds=result.elapsed_seconds,
        )

    def compile(self, work_dir: Path) -> CommandResult:
        """Compila um bug já checked out."""
        if not work_dir.exists():
            self.logger.error("Diretório não existe: %s", work_dir)
            return CommandResult(
                success=False,
                stdout="",
                stderr="Diretório não existe",
                exit_code=-1,
                elapsed_seconds=0.0,
            )

        if self.config.env_type == "local" and "perl" in self.defects4j_cmd:
            args = self.defects4j_cmd.split() + ["compile"]
        else:
            args = [self.defects4j_cmd, "compile"]

        self.logger.info("Compilando %s", work_dir.name)
        result = self._run_command(args, cwd=work_dir, timeout=600)

        if result.success:
            self.logger.info("Compilação concluída em %.2fs", result.elapsed_seconds)
        else:
            self.logger.error(
                "Compilação falhou (exit %d): %s",
                result.exit_code,
                result.stderr[:200],
            )

        return result

    def export(
        self,
        work_dir: Path,
        property_name: str,
        output_file: Optional[Path] = None,
    ) -> Tuple[bool, str]:
        """Exporta uma propriedade do Defects4J."""
        if not work_dir.exists():
            self.logger.error("Diretório não existe: %s", work_dir)
            return False, ""

        if self.config.env_type == "local" and "perl" in self.defects4j_cmd:
            args = self.defects4j_cmd.split() + ["export", "-p", property_name]
        else:
            args = [self.defects4j_cmd, "export", "-p", property_name]

        if output_file:
            args.extend(["-o", str(output_file)])

        self.logger.debug("Exportando propriedade: %s", property_name)
        result = self._run_command(args, cwd=work_dir, timeout=60)

        if result.success:
            if output_file and output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = result.stdout.strip()

            self.logger.debug("Export %s OK: %d chars", property_name, len(content))
            return True, content

        self.logger.error(
            "Export %s falhou: %s",
            property_name,
            result.stderr[:200],
        )
        return False, ""

    def info(self, project: str, bug_id: Optional[int] = None) -> CommandResult:
        """Obtém informações sobre um projeto ou bug."""
        if self.config.env_type == "local" and "perl" in self.defects4j_cmd:
            args = self.defects4j_cmd.split() + ["info", "-p", project]
        else:
            args = [self.defects4j_cmd, "info", "-p", project]

        if bug_id is not None:
            args.extend(["-b", str(bug_id)])

        return self._run_command(args, timeout=30)
