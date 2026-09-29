#!/usr/bin/env python3
"""Gera features.csv processando um bug por vez e liberando o checkout em seguida.

O prepare_dataset.py faz o checkout de todos os bugs antes de enumerar, o que
exige manter mais de 300 workspaces compilados em disco ao mesmo tempo (dezenas
de gigabytes). Este script faz o mesmo trabalho em fluxo: checkout, compilação,
metadados, enumeração e descarte do workspace, bug por bug. Assim a base inteira
é regenerada com alguns gigabytes de pico.

Também registra, para cada bug, se ele entrou na base e por que não entrou,
em data/processed/dataset_build_report.csv. Sem esse registro não há como
distinguir um bug ausente por falha real de checkout de um bug simplesmente
esquecido.

A escrita é incremental: as linhas de cada bug vão para o CSV assim que ficam
prontas, e uma execução interrompida é retomada de onde parou, pulando os bugs
que já estão no arquivo. Numa base em que um único bug do Mockito leva mais de
uma hora para compilar, acumular tudo em memória até o fim significa perder
horas de trabalho a cada interrupção. Bugs que falharam não são pulados: a
falha pode ter sido transitória, e repeti-la é o que distingue erro real de
acidente.

Uso (dentro do container):
  python3 scripts/build_dataset_streaming.py
  python3 scripts/build_dataset_streaming.py --projects Lang,Chart
  python3 scripts/build_dataset_streaming.py --keep-checkouts
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import (
    DATA_INTERMEDIATE_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    PROJECTS,
)
from src.defects4j.checkout import BugInfo, CheckoutManager
from src.defects4j.metadata_exporter import MetadataExporter
from src.defects4j.wrapper import Defects4JWrapper
from src.feature_engineering.engineer import FeatureEngineer
from src.feature_engineering.test_enumerator import TestEnumerator
from src.utils.environment import validate_environment
from src.utils.helpers import Timer, format_duration
from src.utils.logger import setup_logger

REPORT_PATH = DATA_PROCESSED_DIR / "dataset_build_report.csv"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", default=",".join(PROJECTS))
    parser.add_argument(
        "--keep-checkouts",
        action="store_true",
        help="Não apagar o workspace de cada bug após enumerar",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Processar no máximo N bugs por projeto (para testes rápidos)",
    )
    parser.add_argument(
        "--bugs",
        default=None,
        help="Faixa de bugs a processar, ex.: 1-53. Permite dividir um projeto "
             "grande entre vários processos.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Sufixo nos arquivos de saída, para rodar projetos em paralelo "
             "(ex.: --suffix _Lang gera features_Lang.csv). Os parciais são "
             "juntados depois por scripts/merge_dataset_parts.py.",
    )
    return parser.parse_args()


def load_done(path: Path) -> set:
    """Pares (project, bug) que já estão no CSV parcial."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["project", "bug"])
    return set(zip(df["project"], df["bug"].astype(int)))


def append_csv(path: Path, df: pd.DataFrame) -> None:
    """Acrescenta linhas ao CSV, escrevendo o cabeçalho só na criação."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_report(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
        if new:
            writer.writeheader()
        writer.writerow(entry)


def main() -> int:
    args = parse_arguments()
    logger = setup_logger()
    projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    suffix = args.suffix
    intermediate_path = DATA_INTERMEDIATE_DIR / f"intermediate{suffix}.csv"
    report_path = DATA_PROCESSED_DIR / f"dataset_build_report{suffix}.csv"

    try:
        config = validate_environment()
    except RuntimeError as exc:
        logger.error("Erro de ambiente: %s", exc)
        return 1

    checkout_manager = CheckoutManager(config, DATA_RAW_DIR / (suffix.lstrip("_") or "all"))
    wrapper = Defects4JWrapper(config)
    exporter = MetadataExporter(wrapper)
    enumerator = TestEnumerator(config)
    engineer = FeatureEngineer(config)

    done = load_done(intermediate_path)
    if done:
        logger.info("Retomando: %d bug(s) já no arquivo parcial", len(done))

    n_ok = 0
    n_failed = 0
    n_incomplete = 0
    timer = Timer()
    timer.__enter__()

    for project in projects:
        bug_ids = checkout_manager.read_active_bugs(project)
        if args.bugs:
            low, _, high = args.bugs.partition("-")
            low, high = int(low), int(high or low)
            bug_ids = [b for b in bug_ids if low <= b <= high]
        if args.limit:
            bug_ids = bug_ids[: args.limit]
        logger.info("=" * 80)
        logger.info("%s: %d bugs ativos", project, len(bug_ids))
        logger.info("=" * 80)

        for i, bug_id in enumerate(bug_ids, 1):
            tag = f"{project}-{bug_id}"
            logger.info("[%s] %d/%d", project, i, len(bug_ids))

            if (project, bug_id) in done:
                logger.info("%s: já processado, pulando", tag)
                n_ok += 1
                continue

            entry = {
                "project": project,
                "bug": bug_id,
                "status": "",
                "reason": "",
                "n_tests": 0,
                "n_triggers_expected": 0,
                "n_triggers_labeled": 0,
            }

            success, work_dir = checkout_manager.checkout_and_compile(project, bug_id)
            if not success:
                entry.update(
                    status="falhou",
                    reason=checkout_manager.last_failure or "checkout ou compilação",
                )
                append_report(report_path, entry)
                n_failed += 1
                logger.error("%s: checkout/compilação falhou", tag)
                shutil.rmtree(work_dir, ignore_errors=True)
                continue

            metadata = exporter.export_bug_metadata(
                BugInfo(project, bug_id, work_dir), save_files=False
            )
            if metadata is None:
                entry.update(status="falhou", reason="export de metadados")
                append_report(report_path, entry)
                n_failed += 1
                logger.error("%s: export de metadados falhou", tag)
                if not args.keep_checkouts:
                    shutil.rmtree(work_dir, ignore_errors=True)
                continue

            methods = enumerator.enumerate_bug_methods(metadata)
            entry["n_triggers_expected"] = len(metadata.trigger_tests)

            if not methods:
                entry.update(status="falhou", reason="nenhum método de teste enumerado")
                append_report(report_path, entry)
                n_failed += 1
                logger.error("%s: nenhum método enumerado", tag)
                if not args.keep_checkouts:
                    shutil.rmtree(work_dir, ignore_errors=True)
                continue

            df_bug = engineer.build_intermediate_table([metadata], {tag: methods})

            labeled = int(df_bug["is_trigger"].sum())
            entry.update(
                status="ok",
                n_tests=len(df_bug),
                n_triggers_labeled=labeled,
            )
            missing = len(metadata.trigger_tests) - labeled
            if missing > 0:
                entry["reason"] = f"{missing} trigger(s) não enumerado(s)"
                n_incomplete += 1

            # Grava antes de seguir: assim uma interrupção custa um bug, não a
            # execução inteira.
            append_csv(intermediate_path, df_bug)
            append_report(report_path, entry)
            n_ok += 1

            if not args.keep_checkouts:
                shutil.rmtree(work_dir, ignore_errors=True)

    timer.__exit__(None, None, None)

    if not intermediate_path.exists():
        logger.error("Nenhum bug processado com sucesso")
        return 1

    df_intermediate = pd.read_csv(intermediate_path)

    logger.info("=" * 80)
    logger.info("Concluído em %s", format_duration(timer.elapsed))
    logger.info(
        "%s: %d linhas, %d bugs",
        intermediate_path.name,
        len(df_intermediate),
        df_intermediate.groupby(["project", "bug"]).ngroups,
    )
    logger.info("Bugs processados nesta execução: %d", n_ok)
    logger.info("Bugs que falharam: %d", n_failed)
    logger.info("Bugs com trigger não enumerado: %d", n_incomplete)
    logger.info("Relatório: %s", report_path)
    logger.info("As features saem de scripts/merge_dataset_parts.py")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
