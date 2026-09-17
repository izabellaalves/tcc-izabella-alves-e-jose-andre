#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import DATA_INTERMEDIATE_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.defects4j.checkout import CheckoutManager
from src.defects4j.metadata_exporter import MetadataExporter
from src.defects4j.wrapper import Defects4JWrapper
from src.feature_engineering.engineer import FeatureEngineer
from src.feature_engineering.test_enumerator import TestEnumerator
from src.utils.environment import validate_environment
from src.utils.helpers import Timer, format_duration
from src.utils.logger import setup_logger

DEFAULT_MISSING_BUGS = {
    "Lang": [3, 28, 40],
    "Mockito": [16, 17, 34, 35, 36, 37, 38],
    "Math": [1, 36, 100, 101],
}


def parse_bug_specs(specs: List[str]) -> Dict[str, List[int]]:
    bugs: Dict[str, List[int]] = {}

    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Formato inválido: {spec!r}. Use Projeto:id1,id2")
        project, ids_part = spec.split(":", 1)
        project = project.strip()
        ids: List[int] = []
        for token in ids_part.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start, end = token.split("-", 1)
                ids.extend(range(int(start), int(end) + 1))
            else:
                ids.append(int(token))
        bugs[project] = sorted(set(ids))

    return bugs


def bugs_to_pairs(bugs: Dict[str, List[int]]) -> List[Tuple[str, int]]:
    return [(project, bug_id) for project, ids in sorted(bugs.items()) for bug_id in ids]


def drop_bug_rows(df: pd.DataFrame, pairs: Set[Tuple[str, int]]) -> pd.DataFrame:
    if not pairs:
        return df
    mask = df.apply(lambda r: (r["project"], int(r["bug"])) in pairs, axis=1)
    return df.loc[~mask].copy()


def filter_pairs_not_in_csv(
    df: pd.DataFrame, pairs: Set[Tuple[str, int]]
) -> Set[Tuple[str, int]]:
    existing = set(zip(df["project"], df["bug"].astype(int)))
    return {pair for pair in pairs if pair not in existing}


def merge_intermediate(
    existing_path: Path,
    new_rows: pd.DataFrame,
    replaced_pairs: Set[Tuple[str, int]],
) -> pd.DataFrame:
    if existing_path.exists():
        df_existing = pd.read_csv(existing_path)
        df_existing = drop_bug_rows(df_existing, replaced_pairs)
        df_merged = pd.concat([df_existing, new_rows], ignore_index=True)
    else:
        df_merged = new_rows.copy()

    return df_merged.sort_values(
        ["project", "bug", "test_class", "test_method"]
    ).reset_index(drop=True)


def recalculate_history_for_projects(
    df: pd.DataFrame,
    projects: Set[str],
    engineer: FeatureEngineer,
) -> pd.DataFrame:
    df = df.copy()

    for project in sorted(projects):
        mask = df["project"] == project
        if not mask.any():
            continue

        df_project = df.loc[
            mask, ["project", "bug", "test_class", "test_method", "label"]
        ].copy()
        df_project["is_trigger"] = df_project["label"]

        history = engineer.calculate_history_feature(df_project)
        df.loc[mask, "history"] = history.reindex(df.loc[mask].index).values

    return df


def merge_features(
    existing_path: Path,
    new_features: pd.DataFrame,
    replaced_pairs: Set[Tuple[str, int]],
    projects_to_recalc_history: Set[str],
    engineer: FeatureEngineer,
) -> pd.DataFrame:
    df_existing = pd.read_csv(existing_path)
    before_projects = set(df_existing["project"].unique())

    df_existing = drop_bug_rows(df_existing, replaced_pairs)
    df_final = pd.concat([df_existing, new_features], ignore_index=True)
    df_final = recalculate_history_for_projects(
        df_final, projects_to_recalc_history, engineer
    )

    df_final = df_final.sort_values(
        ["project", "bug", "test_class", "test_method"]
    ).reset_index(drop=True)

    after_projects = set(df_final["project"].unique())
    if before_projects - after_projects:
        raise RuntimeError(
            "Merge removeu projetos do features.csv: "
            f"{sorted(before_projects - after_projects)}"
        )

    return df_final


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adiciona bugs ausentes ao dataset sem reprocessar tudo",
    )
    parser.add_argument(
        "--bugs",
        nargs="*",
        help="Bugs no formato Projeto:id1,id2 (ex.: Lang:3,28,40 Math:1,36)",
    )
    parser.add_argument(
        "--skip-checkout",
        action="store_true",
        help="Reutilizar checkout existente se já compilado",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista os bugs que seriam processados",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    logger = setup_logger()

    bugs_map = parse_bug_specs(args.bugs) if args.bugs else DEFAULT_MISSING_BUGS
    requested_pairs = set(bugs_to_pairs(bugs_map))

    logger.info("=" * 80)
    logger.info("Adição incremental de bugs ausentes")
    logger.info("=" * 80)
    for project, ids in sorted(bugs_map.items()):
        logger.info("  %s: %s", project, ids)
    logger.info("Total solicitado: %d bug(s)", len(requested_pairs))

    features_path = DATA_PROCESSED_DIR / "features.csv"
    intermediate_path = DATA_INTERMEDIATE_DIR / "intermediate.csv"

    if not features_path.exists():
        logger.error("Arquivo não encontrado: %s", features_path)
        return 1

    df_existing = pd.read_csv(features_path)
    pairs_to_add = filter_pairs_not_in_csv(df_existing, requested_pairs)
    already_present = requested_pairs - pairs_to_add

    if already_present:
        logger.info("Já presentes no CSV (%d):", len(already_present))
        for project, bug_id in sorted(already_present):
            logger.info("  - %s-%s", project, bug_id)

    if not pairs_to_add:
        logger.info("Nenhum bug novo para adicionar.")
        return 0

    logger.info("Bugs a processar (%d):", len(pairs_to_add))
    for project, bug_id in sorted(pairs_to_add):
        logger.info("  - %s-%s", project, bug_id)

    if args.dry_run:
        return 0

    try:
        config = validate_environment()
    except RuntimeError as exc:
        logger.error("Erro de ambiente: %s", exc)
        return 1

    checkout_manager = CheckoutManager(config, DATA_RAW_DIR)
    bug_pairs = sorted(pairs_to_add)

    logger.info("")
    logger.info("ETAPA 1: Checkout e compilação")
    with Timer() as t:
        bug_infos = checkout_manager.process_bugs(
            bug_pairs, skip_if_exists=args.skip_checkout
        )

    if not bug_infos:
        logger.error("Nenhum bug foi processado com sucesso")
        return 1

    logger.info(
        "Checkout concluído em %s (%d/%d)",
        format_duration(t.elapsed),
        len(bug_infos),
        len(bug_pairs),
    )

    successful_pairs = {(b.project, b.bug_id) for b in bug_infos}
    failed_pairs = pairs_to_add - successful_pairs
    if failed_pairs:
        logger.warning("Bugs que falharam e não entrarão no merge:")
        for project, bug_id in sorted(failed_pairs):
            logger.warning("  - %s-%s", project, bug_id)

    wrapper = Defects4JWrapper(config)
    metadata_exporter = MetadataExporter(wrapper)
    test_enumerator = TestEnumerator(config)
    feature_engineer = FeatureEngineer(config)

    logger.info("")
    logger.info("ETAPA 2: Metadados e enumeração de testes")
    with Timer() as t:
        metadatas = metadata_exporter.export_multiple_bugs(bug_infos)
        test_methods_dict = test_enumerator.enumerate_multiple_bugs(metadatas)

    if not metadatas:
        logger.error("Nenhum metadado exportado")
        return 1

    total_methods = sum(len(methods) for methods in test_methods_dict.values())
    if total_methods == 0:
        logger.error("Nenhum método de teste enumerado")
        return 1

    logger.info(
        "Metadados/testes em %s — %d bug(s), %d método(s)",
        format_duration(t.elapsed),
        len(metadatas),
        total_methods,
    )

    logger.info("")
    logger.info("ETAPA 3: Tabela intermediária dos novos bugs")
    with Timer() as t:
        df_new_intermediate = feature_engineer.build_intermediate_table(
            metadatas,
            test_methods_dict,
        )

    if df_new_intermediate.empty:
        logger.error("Nenhuma linha intermediária gerada")
        return 1

    added_pairs = {
        (row["project"], int(row["bug"]))
        for _, row in df_new_intermediate[["project", "bug"]].drop_duplicates().iterrows()
    }

    logger.info(
        "Novas linhas intermediárias: %d (%s)",
        len(df_new_intermediate),
        format_duration(t.elapsed),
    )

    logger.info("")
    logger.info("ETAPA 4: Features dos novos bugs")
    with Timer() as t:
        df_new_features = feature_engineer.calculate_features(
            df_new_intermediate, [], {}
        )
        if not feature_engineer.validate_features(df_new_features):
            logger.error("Validação das novas features falhou")
            return 1

    logger.info("Novas linhas de features: %d (%s)", len(df_new_features), format_duration(t.elapsed))

    if intermediate_path.exists():
        logger.info("")
        logger.info("ETAPA 5: Merge em intermediate.csv")
        df_intermediate = merge_intermediate(
            intermediate_path,
            df_new_intermediate,
            added_pairs,
        )
        DATA_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        df_intermediate.to_csv(intermediate_path, index=False)
        logger.info("intermediate.csv atualizado: %d linhas", len(df_intermediate))
    else:
        logger.warning("intermediate.csv não encontrado; pulando atualização")

    logger.info("")
    logger.info("ETAPA 6: Merge seguro em features.csv")
    projects_to_recalc = {project for project, _ in added_pairs}

    with Timer() as t:
        df_features = merge_features(
            features_path,
            df_new_features,
            added_pairs,
            projects_to_recalc,
            feature_engineer,
        )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(features_path, index=False)
    logger.info(
        "features.csv atualizado em %s: %d linhas (antes %d)",
        format_duration(t.elapsed),
        len(df_features),
        len(df_existing),
    )

    logger.info("")
    logger.info("Resumo final:")
    for project in sorted(bugs_map):
        bugs_in_file = sorted(
            df_features.loc[df_features["project"] == project, "bug"].unique()
        )
        requested = bugs_map[project]
        added = [b for b in requested if b in bugs_in_file]
        missing = [b for b in requested if b not in bugs_in_file]
        logger.info("  %s: %d bugs no CSV", project, len(bugs_in_file))
        if added:
            logger.info("    presentes: %s", added)
        if missing:
            logger.warning("    ainda ausentes: %s", missing)

    logger.info("")
    logger.info("Projetos no CSV: %s", sorted(df_features["project"].unique()))
    logger.info("Concluído com sucesso")
    return 0


if __name__ == "__main__":
    sys.exit(main())
