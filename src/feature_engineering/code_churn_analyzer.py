import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from src.defects4j.metadata_exporter import BugMetadata
from src.defects4j.wrapper import Defects4JWrapper
from src.utils.environment import EnvironmentConfig
from src.utils.logger import get_logger


class CodeChurnAnalyzer:

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.logger = get_logger()
        self.wrapper = Defects4JWrapper(config)
        self.cache: Dict[str, int] = {}

    def _checkout_fixed_version(
        self,
        metadata: BugMetadata,
        fixed_work_dir: Path,
    ) -> bool:
        if fixed_work_dir.exists():
            config_file = fixed_work_dir / ".defects4j.config"
            if config_file.exists():
                self.logger.debug(
                    "%s: versão fixed já existe",
                    metadata,
                )
                return True

        self.logger.info(
            "Fazendo checkout da versão fixed de %s",
            metadata,
        )

        result = self.wrapper.checkout(
            metadata.project,
            metadata.bug_id,
            "f",
            fixed_work_dir,
        )

        if not result.success:
            self.logger.error(
                "Falha ao fazer checkout da versão fixed de %s",
                metadata,
            )
            return False

        return True

    def _calculate_diff_stats(
        self,
        buggy_dir: Path,
        fixed_dir: Path,
        modified_classes: List[str],
    ) -> int:
        if not modified_classes:
            return 0

        total_churn = 0

        try:
            for modified_class in modified_classes:
                class_path = modified_class.replace(".", "/") + ".java"
                
                buggy_file = buggy_dir / class_path
                fixed_file = fixed_dir / class_path
                
                if not buggy_file.exists() and not fixed_file.exists():
                    self.logger.debug(
                        "Arquivo não encontrado: %s",
                        class_path,
                    )
                    continue

                args = [
                    "diff",
                    "-u",
                    str(buggy_file) if buggy_file.exists() else "/dev/null",
                    str(fixed_file) if fixed_file.exists() else "/dev/null",
                ]

                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    encoding="utf-8",
                    errors="replace",
                )

                lines_added = 0
                lines_removed = 0

                for line in result.stdout.split('\n'):
                    if line.startswith('+') and not line.startswith('+++'):
                        lines_added += 1
                    elif line.startswith('-') and not line.startswith('---'):
                        lines_removed += 1

                total_churn += lines_added + lines_removed

            return total_churn

        except Exception as e:
            self.logger.error(
                "Erro ao calcular diff: %s",
                e,
            )
            return 0

    def get_code_churn(
        self,
        metadata: BugMetadata,
    ) -> int:
        bug_key = str(metadata)
        
        if bug_key in self.cache:
            return self.cache[bug_key]

        fixed_work_dir = metadata.work_dir.parent / f"{metadata.project}-{metadata.bug_id}-fixed"

        if not self._checkout_fixed_version(metadata, fixed_work_dir):
            self.logger.warning(
                "%s: não foi possível obter versão fixed",
                metadata,
            )
            self.cache[bug_key] = 0
            return 0

        churn = self._calculate_diff_stats(
            metadata.work_dir,
            fixed_work_dir,
            metadata.modified_classes,
        )

        self.logger.info(
            "%s: code_churn = %d",
            metadata,
            churn,
        )

        self.cache[bug_key] = churn
        return churn

    def clear_cache(self):
        self.cache.clear()
