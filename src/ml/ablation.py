"""Ablação de features do Random Forest, sobre o conjunto de teste.

Responde duas perguntas que a baseline `same_package` levanta:

1. O modelo de três features ganha da heurística `same_package` sozinha?
2. O que sobra do modelo quando `same_package` sai?

Há um detalhe estrutural que a ablação torna visível: `modified_classes_count`
é constante dentro de cada bug, porque conta as classes modificadas do bug e não
do teste. Como o APFD só depende da ordem dos testes DENTRO de um bug, essa
feature tem poder de ordenação zero, por mais alta que seja sua importância no
classificador. Na prática o modelo de três features ordena com duas, e sem
`same_package` sobra apenas `history`, o que reduz o modelo à baseline
History-based.

Uso: python3 -m src.ml.ablation
Gera: results/ablation/apfd_by_bug.csv, results/ablation/summary.csv,
      results/ablation/wilcoxon.csv
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier

from src.metrics.apfd import calculate_apfd
from src.ml.random_forest_pipeline import CSV_PATH, FEATURES, RESULTS_DIR, mask_for

OUTPUT_DIR = RESULTS_DIR / "ablation"
APFD_DECIMALS = 10

# Variantes do modelo. O nome descreve o conjunto de features usado para ordenar.
VARIANTS = {
    "RF 3 features": FEATURES,
    "RF sem same_package": ["history", "modified_classes_count"],
    "RF só same_package": ["same_package"],
    "RF só history": ["history"],
}


def apfd_by_bug(df_test: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """APFD por bug, ordenando por score decrescente e test_method alfabético."""
    df = df_test.assign(_score=scores)
    rows = []
    for (project, bug), group in df.groupby(["project", "bug"]):
        ordered = group.sort_values(["_score", "test_method"], ascending=[False, True])
        rows.append(
            {
                "project": project,
                "bug": bug,
                "apfd": calculate_apfd(ordered["label"].to_numpy()),
            }
        )
    return pd.DataFrame(rows).sort_values(["project", "bug"]).reset_index(drop=True)


def constant_within_bug(df: pd.DataFrame) -> dict:
    """Para cada feature, em quantos bugs ela tem um único valor."""
    grouped = df.groupby(["project", "bug"])[FEATURES].nunique()
    return {feature: int((grouped[feature] == 1).sum()) for feature in FEATURES}


def run() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    with open(RESULTS_DIR / "train_test_split.json") as f:
        split = json.load(f)

    df_train = df[mask_for(df, split["train"])]
    df_test = df[mask_for(df, split["test"])]

    with open(RESULTS_DIR / "rf_hyperparameters.json") as f:
        params = json.load(f)["best_params"]

    n_bugs = df.groupby(["project", "bug"]).ngroups
    print("Features constantes dentro do bug (sem poder de ordenação):")
    for feature, count in constant_within_bug(df).items():
        flag = "  <-- constante em TODOS os bugs" if count == n_bugs else ""
        print(f"  {feature:<24} {count:>3} de {n_bugs} bugs{flag}")

    merged = None
    for name, features in VARIANTS.items():
        model = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, **params
        ).fit(df_train[features], df_train["label"])
        proba = pd.Series(
            model.predict_proba(df_test[features])[:, 1], index=df_test.index
        )
        result = apfd_by_bug(df_test, proba).rename(columns={"apfd": name})
        merged = result if merged is None else merged.merge(result, on=["project", "bug"])

    # As baselines entram para situar as variantes.
    for name, filename, column in [
        ("Random", "random_baseline_apfd.csv", "apfd_mean"),
        ("History-based", "history_baseline_apfd.csv", "apfd"),
        ("same_package", "same_package_baseline_apfd.csv", "apfd"),
    ]:
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        base = pd.read_csv(path)[["project", "bug", column]].rename(
            columns={column: name}
        )
        merged = merged.merge(base, on=["project", "bug"], how="left")

    strategies = [c for c in merged.columns if c not in ("project", "bug")]
    merged[strategies] = merged[strategies].round(APFD_DECIMALS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_DIR / "apfd_by_bug.csv", index=False)

    summary = (
        merged[strategies]
        .agg(["mean", "median", "std"])
        .T.round(4)
        .sort_values("mean", ascending=False)
        .reset_index()
        .rename(columns={"index": "strategy"})
    )
    summary.insert(1, "n_bugs", len(merged))
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print(f"\nAPFD sobre {len(merged)} bugs de teste:")
    print(summary.to_string(index=False))

    rows = []
    for a, b in combinations(strategies, 2):
        if merged[a].equals(merged[b]):
            rows.append(
                {
                    "comparison": f"{a} vs {b}",
                    "p_value": float("nan"),
                    "significant_p<0.05": False,
                    "identical": True,
                    "wins_a": 0,
                    "wins_b": 0,
                    "ties": len(merged),
                }
            )
            continue
        _, p = wilcoxon(merged[a], merged[b])
        rows.append(
            {
                "comparison": f"{a} vs {b}",
                "p_value": p,
                "significant_p<0.05": p < 0.05,
                "identical": False,
                "wins_a": int((merged[a] > merged[b]).sum()),
                "wins_b": int((merged[b] > merged[a]).sum()),
                "ties": int((merged[a] == merged[b]).sum()),
            }
        )
    tests = pd.DataFrame(rows)
    tests.to_csv(OUTPUT_DIR / "wilcoxon.csv", index=False)

    print("\nWilcoxon pareado:")
    for r in tests.itertuples(index=False):
        verdict = (
            "IDÊNTICOS"
            if r.identical
            else ("SIGNIFICATIVA" if getattr(r, "_2") else "não significativa")
        )
        p = "     -" if r.identical else f"{r.p_value:.4f}"
        print(
            f"  {r.comparison:<42} p={p}  "
            f"{r.wins_a:>2}x{r.wins_b:<2} ({r.ties:>2} empates)  {verdict}"
        )

    print(f"\nSalvo em {OUTPUT_DIR}/")
    return merged


if __name__ == "__main__":
    run()
