"""Baseline History-Based: prioriza testes com maior histórico de detecção de falhas.

Ordenação determinística: history decrescente, empates por test_method alfabético.
Uso: python3 -m src.baselines.history_baseline
Gera results/history_baseline_apfd.csv (uma linha por bug).
"""

from pathlib import Path

import pandas as pd

from src.metrics.apfd import calculate_apfd

CSV_PATH = "data/processed/features.csv"
OUTPUT_PATH = "results/history_baseline_apfd.csv"


def run(csv_path: str = CSV_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rows = []
    for (project, bug), group in df.groupby(["project", "bug"]):
        ordered = group.sort_values(
            ["history", "test_method"], ascending=[False, True]
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

    print(f"Resultados salvos em {output_path} ({len(results)} bugs, ordenação determinística)")
    print("\nResumo agregado do APFD:")
    print(f"  média geral:   {results['apfd'].mean():.4f}")
    print(f"  mediana:       {results['apfd'].median():.4f}")
    print(f"  desvio padrão: {results['apfd'].std(ddof=1):.4f}")
    print(f"  mínimo:        {results['apfd'].min():.4f}")
    print(f"  máximo:        {results['apfd'].max():.4f}")
    print("\nPor projeto:")
    print(results.groupby("project")["apfd"].agg(["mean", "median", "std"]).round(4).to_string())
    return results


if __name__ == "__main__":
    run()
