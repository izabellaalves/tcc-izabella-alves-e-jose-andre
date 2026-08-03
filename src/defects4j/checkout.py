"""Checkout e compilação de bugs do Defects4J."""

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from src.defects4j.wrapper import Defects4JWrapper
from src.utils.environment import EnvironmentConfig
from src.utils.helpers import Timer
from src.utils.logger import BugProcessingLogger, get_logger


@dataclass
class BugInfo:
    """Informações de um bug."""

    project: str
    bug_id: int
    work_dir: Path

    def __str__(self):
        return f"{self.project}-{self.bug_id}"


class CheckoutManager:
    """Gerencia checkout e compilação de bugs."""

    def __init__(self, config: EnvironmentConfig, data_dir: Path = Path("data")):
        self.config = config
        self.data_dir = data_dir
        self.wrapper = Defects4JWrapper(config)
        self.logger = get_logger()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def read_active_bugs(self, project: str) -> List[int]:
        """Lê IDs de bugs ativos de um projeto."""
        csv_path = (
            self.config.defects4j_path
            / "framework"
            / "projects"
            / project
            / "active-bugs.csv"
        )

        if not csv_path.exists():
            self.logger.error("Arquivo de bugs não encontrado: %s", csv_path)
            return []

        bug_ids = []

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bug_ids.append(int(row["bug.id"]))

            self.logger.info(
                "Projeto %s: %d bugs ativos encontrados",
                project,
                len(bug_ids),
            )
            return sorted(bug_ids)

        except Exception as e:
            self.logger.error("Erro ao ler %s: %s", csv_path, e)
            return []

    def get_bug_work_dir(self, project: str, bug_id: int) -> Path:
        return self.data_dir / project / f"{project}-{bug_id}"

    def is_bug_checked_out(self, work_dir: Path) -> bool:
        if not work_dir.exists():
            return False
        return (work_dir / ".defects4j.config").exists()

    def checkout_and_compile(
        self,
        project: str,
        bug_id: int,
        skip_if_exists: bool = False,
    ) -> Tuple[bool, Path]:
        """Faz checkout e compilação de um bug."""
        work_dir = self.get_bug_work_dir(project, bug_id)

        with BugProcessingLogger(self.logger, project, bug_id) as bug_logger:
            if skip_if_exists and self.is_bug_checked_out(work_dir):
                bug_logger.info("Checkout já existe, pulando")
                return True, work_dir

            if work_dir.exists():
                bug_logger.info("Removendo checkout anterior")
                shutil.rmtree(work_dir, ignore_errors=True)

            bug_logger.info("Iniciando checkout")
            with Timer() as t:
                result = self.wrapper.checkout(project, bug_id, "b", work_dir)

            if not result.success:
                bug_logger.error("Checkout falhou após %.2fs", t.elapsed)
                bug_logger.error("Stderr: %s", result.stderr[:500])
                return False, work_dir

            bug_logger.info("Checkout concluído em %.2fs", t.elapsed)

            bug_logger.info("Iniciando compilação")
            with Timer() as t:
                result = self.wrapper.compile(work_dir)

            if not result.success:
                bug_logger.error("Compilação falhou após %.2fs", t.elapsed)
                bug_logger.error("Stderr: %s", result.stderr[:500])
                return False, work_dir

            bug_logger.info("Compilação concluída em %.2fs", t.elapsed)
            return True, work_dir

    def process_project_bugs(
        self,
        project: str,
        skip_if_exists: bool = False,
    ) -> List[BugInfo]:
        """Processa todos os bugs ativos de um projeto."""
        self.logger.info("Processando projeto: %s", project)

        bug_ids = self.read_active_bugs(project)
        if not bug_ids:
            self.logger.error("Nenhum bug encontrado para projeto %s", project)
            return []

        self.logger.info("Total de bugs a processar: %d", len(bug_ids))

        successful_bugs = []
        failed_bugs = []

        for i, bug_id in enumerate(bug_ids, 1):
            self.logger.info("Progresso: %d/%d", i, len(bug_ids))

            success, work_dir = self.checkout_and_compile(
                project,
                bug_id,
                skip_if_exists=skip_if_exists,
            )

            if success:
                successful_bugs.append(BugInfo(project, bug_id, work_dir))
            else:
                failed_bugs.append((project, bug_id))

        self.logger.info("Resumo %s: sucesso %d/%d", project, len(successful_bugs), len(bug_ids))

        if failed_bugs:
            self.logger.warning("Falharam %d bugs:", len(failed_bugs))
            for proj, bid in failed_bugs:
                self.logger.warning("  - %s-%s", proj, bid)

        return successful_bugs

    def process_multiple_projects(
        self,
        projects: List[str],
        skip_if_exists: bool = False,
    ) -> List[BugInfo]:
        """Processa múltiplos projetos."""
        all_bugs = []

        for project in projects:
            all_bugs.extend(self.process_project_bugs(project, skip_if_exists))

        self.logger.info("Total de bugs processados: %d", len(all_bugs))
        self.logger.info("Projetos: %s", ", ".join(projects))

        return all_bugs
