"""Consolida os resultados de APFD das três estratégias nos 26 bugs de teste.

Uso: python3 -m src.metrics.consolidate_results
Gera: results/apfd_long_format.csv, results/descriptive_statistics.csv,
      results/wins_by_bug.csv, results/statistical_tests.csv
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path("results")

STRATEGIES = {
    "Random": ("random_baseline_apfd.csv", "apfd_mean"),
    "History-based": ("history_baseline_apfd.csv", "apfd"),
    "Random Forest": ("random_forest_apfd.csv", "apfd"),
}


def load_long() -> pd.DataFrame:
    with open(RESULTS_DIR / "train_test_split.json") as f:
        split = json.load(f)
    test_pairs = {(proj, bug) for proj, bugs in split["test"].items() for bug in bugs}

    frames = []
    for strategy, (filename, apfd_col) in STRATEGIES.items():
        df = pd.read_csv(RESULTS_DIR / filename)
        df = df[[("project"), "bug", apfd_col, "n_tests", "n_trigger_tests"]]
        df = df.rename(columns={apfd_col: "apfd"})
        df = df[df.apply(lambda r: (r["project"], r["bug"]) in test_pairs, axis=1)]
        df.insert(0, "strategy", strategy)
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)

    # Sanidade: mesmos 26 bugs nas três estratégias, sem NaN
    per_strategy = long_df.groupby("strategy").apply(
        lambda g: set(zip(g["project"], g["bug"])), include_groups=False)
    assert all(bugs == test_pairs for bugs in per_strategy), \
        "Bugs de teste divergem entre estratégias"
    assert not long_df["apfd"].isnull().any(), "APFD nulo encontrado no conjunto de teste"
    assert len(long_df) == 3 * len(test_pairs)

    long_df.to_csv(RESULTS_DIR / "apfd_long_format.csv", index=False)
    return long_df


def descriptive_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    def stats_for(df, scope):
        s = df.groupby("strategy")["apfd"].agg(
            mean="mean", median="median", std="std", min="min",
            q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
            max="max", n_bugs="count").round(4).reset_index()
        s.insert(0, "scope", scope)
        return s

    stats = pd.concat([stats_for(long_df, "Geral")] +
                      [stats_for(long_df[long_df["project"] == p], p)
                       for p in ("Lang", "Chart")], ignore_index=True)
    stats.to_csv(RESULTS_DIR / "descriptive_statistics.csv", index=False)
    return stats


def wins_by_bug(long_df: pd.DataFrame) -> tuple:
    wide = long_df.pivot(index=["project", "bug"], columns="strategy", values="apfd")
    names = list(STRATEGIES)
    best = wide.apply(lambda r: " / ".join(n for n in names if r[n] == r.max()), axis=1)
    worst = wide.apply(lambda r: " / ".join(n for n in names if r[n] == r.min()), axis=1)
    out = wide.round(4).assign(best_strategy=best, worst_strategy=worst).reset_index()
    out.to_csv(RESULTS_DIR / "wins_by_bug.csv", index=False)
    win_counts = best.value_counts()
    return out, win_counts


def statistical_tests(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot(index=["project", "bug"], columns="strategy", values="apfd")
    rows = []
    for a, b in combinations(STRATEGIES, 2):
        stat, p = wilcoxon(wide[a], wide[b])
        rows.append({"comparison": f"{a} vs {b}", "wilcoxon_statistic": stat,
                     "p_value": p, "significant_p<0.05": p < 0.05,
                     "n_pairs": len(wide)})
    tests = pd.DataFrame(rows)
    tests.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)
    return tests


def run():
    long_df = load_long()
    n_bugs = long_df.groupby("strategy").size().iloc[0]
    print(f"Formato longo salvo em results/apfd_long_format.csv "
          f"({n_bugs} bugs x 3 estratégias, sem NaN, bugs idênticos nas três)")

    print("\n=== Estatísticas descritivas do APFD (results/descriptive_statistics.csv) ===")
    print(descriptive_stats(long_df).to_string(index=False))

    print("\n=== Testes de Wilcoxon pareados (results/statistical_tests.csv) ===")
    tests = statistical_tests(long_df)
    for _, r in tests.iterrows():
        sig = "SIGNIFICATIVA" if r["significant_p<0.05"] else "não significativa"
        print(f"  {r['comparison']:<38} W={r['wilcoxon_statistic']:.1f}  "
              f"p={r['p_value']:.4f}  -> {sig} (p<0.05)")

    _, win_counts = wins_by_bug(long_df)
    print("\n=== Vitórias por bug — maior APFD (results/wins_by_bug.csv) ===")
    print(win_counts.to_string())


if __name__ == "__main__":
    run()
