"""Exportação e parsing de metadados do Defects4J."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from src.defects4j.checkout import BugInfo
from src.defects4j.wrapper import Defects4JWrapper
from src.utils.helpers import parse_defects4j_list, parse_test_method_signature
from src.utils.logger import get_logger


@dataclass
class BugMetadata:
    """Metadados de um bug."""

    project: str
    bug_id: int
    work_dir: Path
    trigger_tests: Set[str]
    relevant_test_classes: Set[str]
    modified_classes: List[str]
    test_bin_dir: Optional[Path]

    def __str__(self):
        return f"{self.project}-{self.bug_id}"

    @property
    def modified_classes_count(self) -> int:
        return len(self.modified_classes)


class MetadataExporter:
    """Exporta e parseia metadados do Defects4J."""

    def __init__(self, wrapper: Defects4JWrapper):
        self.wrapper = wrapper
        self.logger = get_logger()

    def export_property(self, work_dir: Path, property_name: str) -> Optional[str]:
        success, content = self.wrapper.export(work_dir, property_name)
        if success:
            return content

        self.logger.warning(
            "Falha ao exportar %s de %s",
            property_name,
            work_dir.name,
        )
        return None

    def save_metadata_files(
        self,
        work_dir: Path,
        metadata_dir: Optional[Path] = None,
    ) -> bool:
        """Exporta e salva arquivos de metadados no disco."""
        if metadata_dir is None:
            metadata_dir = work_dir

        metadata_dir.mkdir(parents=True, exist_ok=True)

        properties = [
            "tests.trigger",
            "tests.relevant",
            "classes.modified",
            "dir.bin.tests",
        ]

        success_count = 0

        for prop in properties:
            output_file = metadata_dir / prop
            success, _ = self.wrapper.export(work_dir, prop, output_file)

            if success:
                success_count += 1
                self.logger.debug("Exportado: %s -> %s", prop, output_file)
            else:
                self.logger.warning("Falha ao exportar: %s", prop)

        return success_count == len(properties)

    def parse_tests_trigger(self, content: str) -> Set[str]:
        lines = parse_defects4j_list(content)
        trigger_tests = set()

        for line in lines:
            try:
                class_name, method_name = parse_test_method_signature(line)
                trigger_tests.add(f"{class_name}::{method_name}")
            except ValueError as e:
                self.logger.warning(
                    "Formato inválido em tests.trigger: %s (%s)",
                    line,
                    e,
                )

        return trigger_tests

    def parse_tests_relevant(self, content: str) -> Set[str]:
        return set(parse_defects4j_list(content))

    def parse_classes_modified(self, content: str) -> List[str]:
        return parse_defects4j_list(content)

    def parse_dir_bin_tests(self, content: str, work_dir: Path) -> Optional[Path]:
        dir_path = content.strip()

        if not dir_path:
            self.logger.warning("dir.bin.tests vazio em %s", work_dir.name)
            return None

        bin_dir = Path(dir_path) if Path(dir_path).is_absolute() else work_dir / dir_path

        if not bin_dir.exists():
            self.logger.warning("Diretório de binários não existe: %s", bin_dir)
            return None

        return bin_dir.resolve()

    def export_bug_metadata(self, bug_info: BugInfo) -> Optional[BugMetadata]:
        work_dir = bug_info.work_dir

        self.logger.info("Exportando metadados de %s", bug_info)

        tests_trigger_content = self.export_property(work_dir, "tests.trigger")
        tests_relevant_content = self.export_property(work_dir, "tests.relevant")
        classes_modified_content = self.export_property(work_dir, "classes.modified")
        dir_bin_tests_content = self.export_property(work_dir, "dir.bin.tests")

        if not all([
            tests_trigger_content,
            tests_relevant_content,
            classes_modified_content,
            dir_bin_tests_content,
        ]):
            self.logger.error("Falha ao exportar metadados de %s", bug_info)
            return None

        trigger_tests = self.parse_tests_trigger(tests_trigger_content)
        relevant_test_classes = self.parse_tests_relevant(tests_relevant_content)
        modified_classes = self.parse_classes_modified(classes_modified_content)
        test_bin_dir = self.parse_dir_bin_tests(dir_bin_tests_content, work_dir)

        self.save_metadata_files(work_dir)

        if not trigger_tests:
            self.logger.warning("%s: nenhum teste trigger encontrado", bug_info)
        if not relevant_test_classes:
            self.logger.warning("%s: nenhuma classe relevante encontrada", bug_info)
        if not modified_classes:
            self.logger.warning("%s: nenhuma classe modificada encontrada", bug_info)
        if test_bin_dir is None:
            self.logger.warning("%s: diretório de binários não encontrado", bug_info)

        metadata = BugMetadata(
            project=bug_info.project,
            bug_id=bug_info.bug_id,
            work_dir=work_dir,
            trigger_tests=trigger_tests,
            relevant_test_classes=relevant_test_classes,
            modified_classes=modified_classes,
            test_bin_dir=test_bin_dir,
        )

        self.logger.info(
            "%s: %d triggers, %d relevant classes, %d modified classes",
            bug_info,
            len(trigger_tests),
            len(relevant_test_classes),
            len(modified_classes),
        )

        return metadata

    def export_multiple_bugs(self, bug_infos: List[BugInfo]) -> List[BugMetadata]:
        metadatas = []
        failed_count = 0

        for i, bug_info in enumerate(bug_infos, 1):
            self.logger.info("Exportando metadados: %d/%d", i, len(bug_infos))

            metadata = self.export_bug_metadata(bug_info)
            if metadata:
                metadatas.append(metadata)
            else:
                failed_count += 1
                self.logger.error("Falha ao exportar %s", bug_info)

        self.logger.info(
            "Metadados exportados: %d/%d (%d falharam)",
            len(metadatas),
            len(bug_infos),
            failed_count,
        )

        return metadatas
