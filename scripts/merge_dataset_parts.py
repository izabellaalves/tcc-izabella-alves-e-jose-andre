#!/usr/bin/env python3
"""Junta os intermediate_*.csv parciais e calcula as features de uma vez.

A regeneração roda em paralelo, um processo por projeto (ou por faixa de bugs de
um projeto grande), para não levar as horas que a versão sequencial levaria. Cada
processo escreve intermediate_<sufixo>.csv e dataset_build_report_<sufixo>.csv,
mas nenhum calcula features: `history` acumula ao longo de todos os bugs
anteriores do projeto, então só faz sentido depois que as partes se juntam.

Uso: python3 scripts/merge_dataset_parts.py
Gera: data/processed/features.csv, data/intermediate/intermediate.csv,
      data/processed/dataset_build_report.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import (
    DATA_INTERMEDIATE_DIR,
    DATA_PROCESSED_DIR,
    EXPECTED_BUGS,
    FEATURE_COLUMNS,
    PROJECTS,
)
from src.feature_engineering.engineer import FeatureEngineer
from src.utils.environment import validate_environment
from src.utils.logger import setup_logger

SORT_KEYS = ["project", "bug", "test_class", "test_method"]


def merge(pattern: str, directory: Path, out_name: str) -> pd.DataFrame | None:
    parts = sorted(directory.glob(pattern))
    if not parts:
        print(f"AVISO: nenhum arquivo casa com {directory}/{pattern}")
        return None

    frames = []
    for part in parts:
        df = pd.read_csv(part)
        frames.append(df)
        print(f"  {part.name}: {len(df)} linhas")

    merged = pd.concat(frames, ignore_index=True)
    keys = [k for k in SORT_KEYS if k in merged.columns]
    if keys:
        merged = merged.sort_values(keys).reset_index(drop=True)

    out = directory / out_name
    merged.to_csv(out, index=False)
    print(f"  -> {out} ({len(merged)} linhas)\n")
    return merged


def main() -> int:
    setup_logger()

    print("intermediate:")
    intermediate = merge("intermediate_*.csv", DATA_INTERMEDIATE_DIR, "intermediate.csv")
    if intermediate is None:
        return 1

    print("features (history calculado sobre a base completa):")
    engineer = FeatureEngineer(validate_environment())
    features = engineer.calculate_features(intermediate, [], {})
    if not engineer.validate_features(features):
        print("ERRO: validação das features falhou")
        return 1
    features.to_csv(DATA_PROCESSED_DIR / "features.csv", index=False)
    print(f"  -> {DATA_PROCESSED_DIR / 'features.csv'} ({len(features)} linhas)\n")

    print("relatório de construção:")
    report = merge(
        "dataset_build_report_*.csv", DATA_PROCESSED_DIR, "dataset_build_report.csv"
    )

    if list(features.columns) != FEATURE_COLUMNS:
        print(f"ERRO: colunas inesperadas em features.csv: {list(features.columns)}")
        return 1

    duplicates = features.duplicated(subset=SORT_KEYS).sum()
    if duplicates:
        print(f"ERRO: {duplicates} linha(s) duplicada(s) após o merge")
        return 1

    print("Resumo por projeto:")
    print(f"  {'projeto':<10}{'bugs':>6}{'ativos':>8}{'instâncias':>12}{'triggers':>10}{'sem trigger':>13}")
    for project in PROJECTS:
        d = features[features["project"] == project]
        if d.empty:
            print(f"  {project:<10}{'AUSENTE':>6}")
            continue
        per_bug = d.groupby("bug")["label"].sum()
        print(
            f"  {project:<10}{d['bug'].nunique():>6}{EXPECTED_BUGS[project]:>8}"
            f"{len(d):>12}{int(d['label'].sum()):>10}{int((per_bug == 0).sum()):>13}"
        )

    per_bug_all = features.groupby(["project", "bug"])["label"].sum()
    print(
        f"  {'TOTAL':<10}{features.groupby(['project','bug']).ngroups:>6}"
        f"{sum(EXPECTED_BUGS.values()):>8}{len(features):>12}"
        f"{int(features['label'].sum()):>10}{int((per_bug_all == 0).sum()):>13}"
    )

    if report is not None:
        failed = report[report["status"] != "ok"]
        print(f"\nBugs que falharam: {len(failed)}")
        for row in failed.itertuples(index=False):
            print(f"  {row.project}-{row.bug}: {row.reason}")

        incomplete = report[(report["status"] == "ok") & report["reason"].notna()]
        incomplete = incomplete[incomplete["reason"].astype(str).str.strip() != ""]
        print(f"Bugs com trigger não enumerado: {len(incomplete)}")
        for row in incomplete.itertuples(index=False):
            print(f"  {row.project}-{row.bug}: {row.reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
