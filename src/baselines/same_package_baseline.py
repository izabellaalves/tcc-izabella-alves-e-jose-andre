"""Baseline Same-Package: ordena por same_package decrescente."""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

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

        rf_path = Path("results/random_forest_apfd.csv")
        if rf_path.exists():
            rf = pd.read_csv(rf_path)
            rf = rf[rf.apply(lambda r: (r["project"], r["bug"]) in test_pairs, axis=1)]
            merged = test.merge(
                rf[["project", "bug", "apfd"]],
                on=["project", "bug"],
                suffixes=("_same_package", "_rf"),
            )
            _, p = wilcoxon(merged["apfd_rf"], merged["apfd_same_package"])
            ties = (merged["apfd_rf"] == merged["apfd_same_package"]).sum()
            rf_wins = (merged["apfd_rf"] > merged["apfd_same_package"]).sum()
            sp_wins = (merged["apfd_same_package"] > merged["apfd_rf"]).sum()
            print("\nComparacao com Random Forest (mesmos bugs de teste):")
            print(f"  RF media:            {merged['apfd_rf'].mean():.4f}")
            print(f"  same_package media:  {merged['apfd_same_package'].mean():.4f}")
            print(f"  Wilcoxon p:          {p:.4f}")
            print(f"  RF melhor: {rf_wins}  |  same_package melhor: {sp_wins}  |  empate: {ties}")

    return results


if __name__ == "__main__":
    run()
