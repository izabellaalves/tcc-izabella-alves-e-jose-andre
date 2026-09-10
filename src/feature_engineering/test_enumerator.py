"""Enumeração de métodos de teste via javap."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.defects4j.metadata_exporter import BugMetadata
from src.utils.environment import EnvironmentConfig
from src.utils.logger import get_logger


@dataclass
class TestMethod:
    """Método de teste identificado em uma classe."""

    class_name: str
    method_name: str

    def __str__(self):
        return f"{self.class_name}::{self.method_name}"

    def to_tuple(self) -> Tuple[str, str]:
        return self.class_name, self.method_name


class TestEnumerator:
    """Enumera métodos de teste usando javap."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.logger = get_logger()

    def find_class_file(self, class_name: str, bin_dir: Path) -> Optional[Path]:
        class_path = class_name.replace(".", "/") + ".class"
        class_file = bin_dir / class_path

        if class_file.exists():
            return class_file

        self.logger.debug("Arquivo .class não encontrado: %s", class_file)
        return None

    def run_javap(self, class_name: str, bin_dir: Path) -> Optional[str]:
        try:
            result = subprocess.run(
                ["javap", "-v", "-cp", str(bin_dir), class_name],
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )

            if result.returncode == 0:
                return result.stdout

            self.logger.debug(
                "javap falhou para %s: %s",
                class_name,
                result.stderr[:200],
            )
            return None

        except subprocess.TimeoutExpired:
            self.logger.warning("javap timeout para %s", class_name)
            return None
        except FileNotFoundError:
            self.logger.error("javap não encontrado")
            return None
        except Exception as e:
            self.logger.error("Erro ao executar javap: %s", e)
            return None

    def parse_javap_output(self, javap_output: str, class_name: str) -> List[str]:
        """Extrai métodos anotados com @Test ou com prefixo test."""
        test_methods = []
        lines = javap_output.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            method_match = re.match(
                r"public\s+(?:void|[\w.<>]+)\s+(\w+)\s*\(",
                line,
            )

            if method_match:
                method_name = method_match.group(1)

                if method_name.startswith("test"):
                    test_methods.append(method_name)
                    self.logger.debug("Método de teste encontrado: %s (prefixo test)", method_name)
                else:
                    has_test_annotation = False
                    for j in range(i + 1, min(i + 100, len(lines))):
                        check_line = lines[j].strip()
                        if "RuntimeVisibleAnnotations:" in check_line:
                            for k in range(j + 1, min(j + 10, len(lines))):
                                annotation_line = lines[k].strip()
                                lower_annotation = annotation_line.lower()
                                if ("org/junit" in lower_annotation or "org.junit" in lower_annotation) and "test" in lower_annotation:
                                    has_test_annotation = True
                                    break
                                if not annotation_line or (annotation_line and not annotation_line[0].isdigit() and not annotation_line.startswith("#") and "org" not in lower_annotation):
                                    break
                            break
                        if (check_line.startswith("public ") or check_line.startswith("private ") or check_line.startswith("protected ")) and "(" in check_line:
                            break

                    if has_test_annotation:
                        test_methods.append(method_name)
                        self.logger.debug("Método de teste encontrado: %s (@Test)", method_name)

            i += 1

        return test_methods

    def enumerate_class_methods(
        self,
        class_name: str,
        bin_dir: Path,
    ) -> List[TestMethod]:
        self.logger.debug("Enumerando métodos: %s", class_name)

        if self.find_class_file(class_name, bin_dir) is None:
            self.logger.warning("Classe não encontrada: %s", class_name)
            return []

        javap_output = self.run_javap(class_name, bin_dir)
        if javap_output is None:
            self.logger.warning("Falha ao executar javap para %s", class_name)
            return []

        method_names = self.parse_javap_output(javap_output, class_name)
        test_methods = [TestMethod(class_name, name) for name in method_names]

        if test_methods:
            self.logger.debug(
                "%s: %d métodos encontrados",
                class_name,
                len(test_methods),
            )
        else:
            self.logger.warning("%s: nenhum método de teste encontrado", class_name)

        return test_methods

    def enumerate_bug_methods(self, metadata: BugMetadata) -> List[TestMethod]:
        if metadata.test_bin_dir is None:
            self.logger.error("%s: diretório de binários não disponível", metadata)
            return []

        self.logger.info(
            "Enumerando métodos de teste para %s (%d classes)",
            metadata,
            len(metadata.relevant_test_classes),
        )

        all_test_methods = []
        seen_methods = set()

        for class_name in sorted(metadata.relevant_test_classes):
            methods = self.enumerate_class_methods(class_name, metadata.test_bin_dir)

            for method in methods:
                signature = (method.class_name, method.method_name)
                if signature in seen_methods:
                    self.logger.debug("Método duplicado ignorado: %s", method)
                    continue

                all_test_methods.append(method)
                seen_methods.add(signature)

        self.logger.info(
            "%s: %d métodos de teste únicos encontrados",
            metadata,
            len(all_test_methods),
        )

        return all_test_methods

    def enumerate_multiple_bugs(
        self,
        metadatas: List[BugMetadata],
    ) -> Dict[str, List[TestMethod]]:
        results = {}
        total_methods = 0

        for i, metadata in enumerate(metadatas, 1):
            self.logger.info("Enumerando: %d/%d", i, len(metadatas))

            methods = self.enumerate_bug_methods(metadata)
            results[str(metadata)] = methods
            total_methods += len(methods)

        self.logger.info(
            "Enumeração concluída: %d métodos em %d bugs",
            total_methods,
            len(metadatas),
        )

        return results

    def get_test_methods_as_tuples(
        self,
        test_methods: List[TestMethod],
    ) -> List[Tuple[str, str]]:
        return [method.to_tuple() for method in test_methods]
