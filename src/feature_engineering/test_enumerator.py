"""Enumeração de métodos de teste via javap.

A enumeração precisa cobrir todos os testes que o Defects4J de fato executa,
porque um trigger test que não é enumerado vira um rótulo perdido: o bug fica
com menos label=1 do que deveria, ou com nenhum, e sai da avaliação de APFD.

Três defeitos da versão anterior, que juntos perdiam 42 dos 634 triggers e
deixavam 13 bugs sem nenhum trigger:

1. **Métodos herdados.** `javap` lista só os métodos declarados na própria
   classe. Testes herdados de uma classe-base abstrata (o padrão
   `RealDistributionAbstractTest` em Math, por exemplo) ficavam de fora, embora
   o JUnit os execute pela subclasse. Agora a enumeração sobe a cadeia de
   superclasses e atribui o método à subclasse, que é como o Defects4J nomeia o
   trigger.
2. **Janela fixa para a anotação.** A anotação `@Test` aparece no fim do bloco
   do método, depois de todo o bytecode. Procurar em no máximo 100 linhas
   perdia qualquer método com corpo longo, o que explicava métodos irmãos da
   mesma classe sendo detectados e outros não. Agora o bloco do método é
   delimitado e varrido por inteiro.
3. **Prefixo sensível a maiúsculas.** O teste era `name.startswith("test")`,
   então `TestLang747` (o trigger do Lang-1) não era reconhecido. O prefixo do
   JUnit 3 agora é comparado sem diferenciar maiúsculas.

Além disso, métodos anotados com `@Ignore` são excluídos (o JUnit não os roda,
então não deveriam ocupar posição na ordenação) e classes abstratas não são
enumeradas como se fossem executáveis.
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.defects4j.metadata_exporter import BugMetadata
from src.utils.environment import EnvironmentConfig
from src.utils.logger import get_logger

# Anotações que marcam um método como teste executável.
TEST_ANNOTATIONS = (
    "org.junit.Test",
    "org.junit.jupiter.api.Test",
    "org.testng.annotations.Test",
)
IGNORE_ANNOTATIONS = (
    "org.junit.Ignore",
    "org.junit.jupiter.api.Disabled",
)

# Declaração de membro no corpo da classe: indentação de 2 e termina em ';'.
MEMBER_RE = re.compile(r"^ {2}(?! )(?P<decl>\S.*\((?P<params>[^)]*)\).*);\s*$")
SUPERCLASS_RE = re.compile(r"^\s*super_class:.*//\s*(?P<name>\S+)\s*$")
CLASS_FLAGS_RE = re.compile(r"^\s*flags:.*$")
# Nome da anotação dentro do bloco RuntimeVisibleAnnotations. O '(' opcional
# cobre anotações com argumentos, que o javap quebra em várias linhas:
#     org.junit.Test(
#       expected=class Ljava/io/IOException;
#     )
# Sem ele, @Test(expected = ...) não era reconhecido como teste.
ANNOTATION_RE = re.compile(r"^\s{6,}(?P<name>[\w.$]+)\s*\(?\s*$")
THIS_CLASS_RE = re.compile(r"^\s*this_class:.*//\s*(?P<name>\S+)\s*$", re.MULTILINE)

# Classes por chamada de javap, e tempo limite do lote. O limite de tamanho é
# do tamanho da linha de comando; 200 nomes ficam muito abaixo dele.
JAVAP_BATCH_SIZE = 200
JAVAP_BATCH_TIMEOUT = 600

# Profundidade máxima da cadeia de superclasses percorrida.
MAX_HIERARCHY_DEPTH = 20


@dataclass
class TestMethod:
    """Método de teste identificado em uma classe."""

    class_name: str
    method_name: str

    def __str__(self):
        return f"{self.class_name}::{self.method_name}"

    def to_tuple(self) -> Tuple[str, str]:
        return self.class_name, self.method_name


@dataclass
class MethodInfo:
    """Método declarado em uma classe, como o javap o descreve."""

    name: str
    is_public: bool
    is_static: bool
    has_params: bool
    annotations: Set[str] = field(default_factory=set)

    @property
    def is_test(self) -> bool:
        """Um teste JUnit é público, de instância, sem argumentos."""
        if not self.is_public or self.is_static or self.has_params:
            return False
        if any(a in self.annotations for a in IGNORE_ANNOTATIONS):
            return False
        if any(a in self.annotations for a in TEST_ANNOTATIONS):
            return True
        # Convenção do JUnit 3. Sem diferenciar maiúsculas por causa de casos
        # como NumberUtilsTest::TestLang747, o trigger do Lang-1.
        return self.name.lower().startswith("test")


@dataclass
class ClassInfo:
    """Classe compilada: superclasse e métodos declarados nela."""

    name: str
    superclass: Optional[str]
    is_abstract: bool
    methods: List[MethodInfo]


class TestEnumerator:
    """Enumera métodos de teste usando javap, incluindo os herdados."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.logger = get_logger()
        # javap é caro e classes-base são revisitadas por cada subclasse.
        self._class_cache: Dict[Tuple[str, str], Optional[ClassInfo]] = {}

    # ------------------------------------------------------------------ javap

    def find_class_file(self, class_name: str, bin_dir: Path) -> Optional[Path]:
        class_file = bin_dir / (class_name.replace(".", "/") + ".class")
        if class_file.exists():
            return class_file
        self.logger.debug("Arquivo .class não encontrado: %s", class_file)
        return None

    def run_javap(self, class_names: List[str], bin_dir: Path) -> Optional[str]:
        """Roda javap sobre várias classes de uma vez.

        Uma chamada por classe domina o custo da enumeração: em Chart-2, 73
        classes levam 27s em chamadas separadas e 1s numa só, porque o que pesa
        é subir a JVM, não ler o bytecode.
        """
        if not class_names:
            return None

        try:
            result = subprocess.run(
                ["javap", "-v", "-cp", str(bin_dir), *class_names],
                capture_output=True,
                text=True,
                timeout=JAVAP_BATCH_TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
            # javap devolve código != 0 se QUALQUER classe falhar, mas ainda
            # imprime as que deram certo, então a saída é aproveitada sempre.
            if result.stdout:
                return result.stdout

            self.logger.debug(
                "javap não produziu saída para %d classe(s): %s",
                len(class_names),
                result.stderr[:200],
            )
            return None

        except subprocess.TimeoutExpired:
            self.logger.warning("javap timeout para %d classe(s)", len(class_names))
            return None
        except FileNotFoundError:
            self.logger.error("javap não encontrado")
            return None
        except Exception as e:  # pragma: no cover - proteção do pipeline longo
            self.logger.error("Erro ao executar javap: %s", e)
            return None

    @staticmethod
    def split_javap_output(output: str) -> Dict[str, str]:
        """Separa a saída de um javap em lote, por classe.

        Cada classe começa numa linha 'Classfile ...' e o nome vem do comentário
        de this_class, que é o nome interno (com '/'), mais confiável que o
        caminho do arquivo.
        """
        blocks: Dict[str, str] = {}
        current: List[str] = []

        def flush(lines: List[str]) -> None:
            if not lines:
                return
            text = "\n".join(lines)
            match = THIS_CLASS_RE.search(text)
            if match:
                blocks[match.group("name").replace("/", ".")] = text

        for line in output.splitlines():
            if line.startswith("Classfile "):
                flush(current)
                current = [line]
            else:
                current.append(line)
        flush(current)

        return blocks

    def load_classes(self, class_names: List[str], bin_dir: Path) -> None:
        """Carrega no cache as classes pedidas que ainda não estão lá.

        As classes vão ao javap em lotes, e os nomes que não têm .class no
        diretório são descartados antes para não fazer o lote inteiro falhar.
        """
        pending = [
            name
            for name in dict.fromkeys(class_names)
            if (name, str(bin_dir)) not in self._class_cache
        ]
        if not pending:
            return

        readable = []
        for name in pending:
            if self.find_class_file(name, bin_dir) is None:
                self._class_cache[(name, str(bin_dir))] = None
            else:
                readable.append(name)

        for start in range(0, len(readable), JAVAP_BATCH_SIZE):
            batch = readable[start : start + JAVAP_BATCH_SIZE]
            output = self.run_javap(batch, bin_dir)
            blocks = self.split_javap_output(output) if output else {}

            for name in batch:
                text = blocks.get(name)
                self._class_cache[(name, str(bin_dir))] = (
                    self.parse_javap_output(text, name) if text else None
                )
                if text is None:
                    self.logger.debug("javap não devolveu bloco para %s", name)

    # ----------------------------------------------------------------- parser

    @staticmethod
    def _member_blocks(lines: List[str]) -> List[Tuple[str, str, List[str]]]:
        """Divide o corpo da classe em blocos (declaração, params, linhas).

        O corpo fica entre '{' e o '}' final. Delimitar o bloco de cada membro,
        em vez de olhar um número fixo de linhas adiante, é o que garante achar
        a anotação de métodos com bytecode longo.
        """
        try:
            start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "{")
        except StopIteration:
            return []

        blocks: List[Tuple[str, str, List[str]]] = []
        current: Optional[List] = None

        for line in lines[start + 1:]:
            if line.rstrip() == "}":
                break
            match = MEMBER_RE.match(line)
            if match:
                if current is not None:
                    blocks.append(tuple(current))
                current = [match.group("decl"), match.group("params"), []]
            elif current is not None:
                current[2].append(line)

        if current is not None:
            blocks.append(tuple(current))

        return blocks

    @staticmethod
    def _annotations_in_block(block_lines: List[str]) -> Set[str]:
        """Nomes das anotações visíveis em tempo de execução do bloco."""
        annotations: Set[str] = set()
        inside = False

        for line in block_lines:
            stripped = line.strip()
            if stripped.startswith("RuntimeVisibleAnnotations:"):
                inside = True
                continue
            if inside:
                # Sai ao encontrar outro atributo do método (indentação de 4).
                if stripped and not line.startswith(" " * 6):
                    inside = False
                    continue
                match = ANNOTATION_RE.match(line)
                if match:
                    annotations.add(match.group("name").replace("$", "."))

        return annotations

    def parse_javap_output(self, javap_output: str, class_name: str) -> ClassInfo:
        """Extrai superclasse, flags e métodos declarados da saída do javap."""
        lines = javap_output.splitlines()

        superclass = None
        is_abstract = False
        for line in lines[:40]:
            match = SUPERCLASS_RE.match(line)
            if match:
                superclass = match.group("name").replace("/", ".")
            if CLASS_FLAGS_RE.match(line) and "ACC_ABSTRACT" in line:
                is_abstract = True
            if line.rstrip() == "{":
                break

        if superclass in ("java.lang.Object", class_name):
            superclass = None

        methods = []
        for decl, params, block in self._member_blocks(lines):
            name_match = re.search(r"([\w$]+)\s*\($", decl.split("(")[0] + "(")
            if not name_match:
                continue
            name = name_match.group(1)
            if name == class_name.rsplit(".", 1)[-1] or name in ("<init>", "<clinit>"):
                continue  # construtor

            methods.append(
                MethodInfo(
                    name=name,
                    is_public=decl.startswith("public "),
                    is_static=" static " in f" {decl} ",
                    has_params=bool(params.strip()),
                    annotations=self._annotations_in_block(block),
                )
            )

        return ClassInfo(
            name=class_name,
            superclass=superclass,
            is_abstract=is_abstract,
            methods=methods,
        )

    # ------------------------------------------------------- cadeia de classes

    def load_class(self, class_name: str, bin_dir: Path) -> Optional[ClassInfo]:
        key = (class_name, str(bin_dir))
        if key not in self._class_cache:
            self.load_classes([class_name], bin_dir)
        return self._class_cache.get(key)

    def load_hierarchy(self, class_names: List[str], bin_dir: Path) -> None:
        """Carrega as classes e, em lote, toda a cadeia de superclasses delas.

        Resolver as superclasses em ondas mantém uma chamada de javap por nível
        da hierarquia, em vez de uma por classe.
        """
        self.load_classes(class_names, bin_dir)

        frontier = list(class_names)
        for _ in range(MAX_HIERARCHY_DEPTH):
            supers = []
            for name in frontier:
                info = self._class_cache.get((name, str(bin_dir)))
                if info and info.superclass:
                    if (info.superclass, str(bin_dir)) not in self._class_cache:
                        supers.append(info.superclass)
            if not supers:
                break
            self.load_classes(supers, bin_dir)
            frontier = supers

    def enumerate_class_methods(
        self,
        class_name: str,
        bin_dir: Path,
    ) -> List[TestMethod]:
        """Métodos de teste executáveis da classe, herdados incluídos.

        Os métodos herdados são atribuídos à subclasse, que é como o Defects4J
        nomeia o trigger em tests.trigger.
        """
        self.logger.debug("Enumerando métodos: %s", class_name)

        info = self.load_class(class_name, bin_dir)
        if info is None:
            self.logger.warning("Classe não encontrada ou ilegível: %s", class_name)
            return []

        if info.is_abstract:
            # Uma classe abstrata não é instanciada: seus testes contam pelas
            # subclasses, que aparecem em tests.relevant por conta própria.
            self.logger.debug("%s é abstrata, ignorando", class_name)
            return []

        # Da subclasse para a raiz: a declaração mais específica ganha.
        seen: Dict[str, MethodInfo] = {}
        current = info
        depth = 0

        while current is not None and depth < MAX_HIERARCHY_DEPTH:
            for method in current.methods:
                seen.setdefault(method.name, method)

            if current.superclass is None:
                break
            current = self.load_class(current.superclass, bin_dir)
            depth += 1

        test_methods = [
            TestMethod(class_name, name)
            for name, method in sorted(seen.items())
            if method.is_test
        ]

        if test_methods:
            self.logger.debug(
                "%s: %d métodos de teste (%d declarados na classe)",
                class_name,
                len(test_methods),
                len(info.methods),
            )
        else:
            self.logger.warning("%s: nenhum método de teste encontrado", class_name)

        return test_methods

    # ------------------------------------------------------------------- bugs

    def enumerate_bug_methods(self, metadata: BugMetadata) -> List[TestMethod]:
        if metadata.test_bin_dir is None:
            self.logger.error("%s: diretório de binários não disponível", metadata)
            return []

        self.logger.info(
            "Enumerando métodos de teste para %s (%d classes)",
            metadata,
            len(metadata.relevant_test_classes),
        )

        # Uma passada de javap em lote para todas as classes relevantes e suas
        # superclasses, antes de enumerar qualquer uma.
        self.load_hierarchy(
            sorted(metadata.relevant_test_classes), metadata.test_bin_dir
        )

        all_test_methods = []
        seen_methods = set()

        for class_name in sorted(metadata.relevant_test_classes):
            for method in self.enumerate_class_methods(
                class_name, metadata.test_bin_dir
            ):
                signature = (method.class_name, method.method_name)
                if signature in seen_methods:
                    self.logger.debug("Método duplicado ignorado: %s", method)
                    continue
                all_test_methods.append(method)
                seen_methods.add(signature)

        missed = sorted(metadata.trigger_tests - {str(m) for m in all_test_methods})
        if missed:
            self.logger.warning(
                "%s: %d trigger test(s) NÃO enumerado(s): %s",
                metadata,
                len(missed),
                ", ".join(missed[:5]),
            )

        self.logger.info(
            "%s: %d métodos de teste únicos encontrados", metadata, len(all_test_methods)
        )

        return all_test_methods

    def enumerate_multiple_bugs(
        self,
        metadatas: List[BugMetadata],
    ) -> Dict[str, List[TestMethod]]:
        results = {}
        total_methods = 0
        bugs_missing_triggers = []

        for i, metadata in enumerate(metadatas, 1):
            self.logger.info("Enumerando: %d/%d", i, len(metadatas))

            methods = self.enumerate_bug_methods(metadata)
            results[str(metadata)] = methods
            total_methods += len(methods)
            # Cada bug tem seu próprio diretório de binários, então o cache do
            # bug anterior nunca volta a ser útil.
            self._class_cache.clear()

            if metadata.trigger_tests - {str(m) for m in methods}:
                bugs_missing_triggers.append(str(metadata))

        self.logger.info(
            "Enumeração concluída: %d métodos em %d bugs", total_methods, len(metadatas)
        )
        if bugs_missing_triggers:
            self.logger.warning(
                "%d bug(s) com trigger não enumerado: %s",
                len(bugs_missing_triggers),
                ", ".join(bugs_missing_triggers),
            )
        else:
            self.logger.info("Todos os trigger tests foram enumerados")

        return results

    def get_test_methods_as_tuples(
        self,
        test_methods: List[TestMethod],
    ) -> List[Tuple[str, str]]:
        return [method.to_tuple() for method in test_methods]
