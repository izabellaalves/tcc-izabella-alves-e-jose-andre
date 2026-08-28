"""Baseline Random: prioriza testes em ordem aleatória (30 sementes por bug).

Uso: python3 -m src.baselines.random_baseline
Gera results/random_baseline_apfd.csv (uma linha por bug).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.apfd import calculate_apfd

CSV_PATH = "data/processed/features.csv"
OUTPUT_PATH = "results/random_baseline_apfd.csv"
N_SEEDS = 30


def run(csv_path: str = CSV_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rows = []
    for (project, bug), group in df.groupby(["project", "bug"]):
        labels = group["label"].to_numpy()
        apfds = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(labels)
            apfds.append(calculate_apfd(shuffled))
        rows.append({
            "project": project,
            "bug": bug,
            "apfd_mean": np.mean(apfds),
            "apfd_std": np.std(apfds, ddof=1),
            "n_tests": len(labels),
            "n_trigger_tests": int(labels.sum()),
        })

    results = pd.DataFrame(rows).sort_values(["project", "bug"]).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Resultados salvos em {output_path} ({len(results)} bugs, {N_SEEDS} seeds cada)")
    print("\nResumo agregado do APFD (média por bug):")
    print(f"  média geral:   {results['apfd_mean'].mean():.4f}")
    print(f"  mediana:       {results['apfd_mean'].median():.4f}")
    print(f"  desvio padrão: {results['apfd_mean'].std(ddof=1):.4f}")
    print(f"  mínimo:        {results['apfd_mean'].min():.4f}")
    print(f"  máximo:        {results['apfd_mean'].max():.4f}")
    print("\nPor projeto:")
    print(results.groupby("project")["apfd_mean"].agg(["mean", "median", "std"]).round(4).to_string())
    return results


if __name__ == "__main__":
    run()
