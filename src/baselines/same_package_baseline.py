"""Baseline Same-Package: ordena por same_package decrescente.

Ordenação determinística: same_package decrescente, empates por test_method
alfabético, igual à baseline History.
Uso: python3 -m src.baselines.same_package_baseline
Gera results/same_package_baseline_apfd.csv (uma linha por bug).
"""

import json
from pathlib import Path

import pandas as pd

from src.metrics.apfd import calculate_apfd

CSV_PATH = "data/processed/features.csv"
OUTPUT_PATH = "results/same_package_baseline_apfd.csv"
SPLIT_PATH = "results/train_test_split.json"


def run(csv_path: str = CSV_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rows = []
    for (project, bug), group in df.groupby(["project", "bug"]):
        ordered = group.sort_values(
            ["same_package", "test_method"], ascending=[False, True]
        )
        labels = ordered["label"].to_numpy()
        rows.append({
            "project": project,
            "bug": bug,
            "apfd": calculate_apfd(labels),
            "n_tests": len(labels),
            "n_trigger_tests": int(labels.sum()),
        })

    results = pd.DataFrame(rows).sort_values(["project", "bug"]).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Resultados salvos em {output_path} ({len(results)} bugs)")
    print("\nResumo agregado do APFD (todos os bugs):")
    print(f"  media geral:   {results['apfd'].mean():.4f}")
    print(f"  mediana:       {results['apfd'].median():.4f}")
    print(f"  desvio padrao: {results['apfd'].std(ddof=1):.4f}")

    if Path(SPLIT_PATH).exists():
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        test_pairs = {
            (proj, bug) for proj, bugs in split["test"].items() for bug in bugs
        }
        test = results[
            results.apply(lambda r: (r["project"], r["bug"]) in test_pairs, axis=1)
        ].dropna(subset=["apfd"])

        print(f"\nConjunto de teste ({len(test)} bugs validos):")
        print(f"  media:   {test['apfd'].mean():.4f}")
        print(f"  mediana: {test['apfd'].median():.4f}")
        print(f"  dp:      {test['apfd'].std(ddof=1):.4f}")
        print("\nPor projeto (teste):")
        print(test.groupby("project")["apfd"].agg(["mean", "median", "std"]).round(4).to_string())

        print(
            "\nComparação estatística com as outras estratégias: "
            "rode 'python3 -m src.metrics.consolidate_results'.\n"
            "Os p-valores saem de lá, sobre APFDs arredondados, para que os empates "
            "não dependam do último bit de ponto flutuante."
        )

    return results


if __name__ == "__main__":
    run()
