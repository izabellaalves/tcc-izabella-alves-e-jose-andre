"""Pipeline Random Forest: split por bug, treino com tuning e avaliação (APFD + P/R/F1).

Uso: python3 -m src.ml.random_forest_pipeline [scoring]
     scoring: f1 (default) ou roc_auc — muda a métrica do GridSearchCV e o
     sufixo dos arquivos de saída (ex. random_forest_apfd_rocauc.csv).
Gera: results/train_test_split.json, results/rf_model*.joblib,
      results/rf_hyperparameters*.json, results/random_forest_apfd*.csv
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.metrics.apfd import calculate_apfd

CSV_PATH = "data/processed/features.csv"
RESULTS_DIR = Path("results")

# Seed fixa da metodologia (TCC1) para o split por bug — garante reprodutibilidade.
SPLIT_SEED = 42
FEATURES = ["history", "same_package", "modified_classes_count"]

# Tamanhos alvo do split (definidos na metodologia do TCC1).
# Lang-1 não tem trigger test (APFD indefinido), então entra FIXO no treino;
# os demais 60 bugs de Lang são sorteados em 42 treino + 18 teste.
SPLIT_PLAN = {"Lang": {"train": 43, "test": 18, "fixed_train": [1]},
              "Chart": {"train": 18, "test": 8, "fixed_train": []}}


def make_split(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)
    split = {"seed": SPLIT_SEED, "train": {}, "test": {}}
    for project, plan in SPLIT_PLAN.items():
        bugs = sorted(df.loc[df["project"] == project, "bug"].unique().tolist())
        fixed = plan["fixed_train"]
        pool = [b for b in bugs if b not in fixed]
        shuffled = rng.permutation(pool).tolist()
        n_test = plan["test"]
        split["test"][project] = sorted(int(b) for b in shuffled[:n_test])
        split["train"][project] = sorted(fixed + [int(b) for b in shuffled[n_test:]])
        assert len(split["train"][project]) == plan["train"]
        assert len(split["test"][project]) == plan["test"]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "train_test_split.json", "w") as f:
        json.dump(split, f, indent=2)
    return split


def mask_for(df: pd.DataFrame, split_side: dict) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for project, bugs in split_side.items():
        mask |= (df["project"] == project) & df["bug"].isin(bugs)
    return mask


def train(df_train: pd.DataFrame, scoring: str = "f1", suffix: str = "") -> tuple:
    X, y = df_train[FEATURES], df_train["label"]
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "min_samples_split": [2, 5, 10],
    }
    search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid,
        scoring=scoring,  # métrica robusta ao desbalanceamento (positivos ~1,3%)
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    search.fit(X, y)
    model = search.best_estimator_
    joblib.dump(model, RESULTS_DIR / f"rf_model{suffix}.joblib")
    with open(RESULTS_DIR / f"rf_hyperparameters{suffix}.json", "w") as f:
        json.dump({"best_params": search.best_params_,
                   f"cv_best_{scoring}": search.best_score_,
                   "scoring": scoring, "cv": "StratifiedKFold(5, shuffle, seed=42)",
                   "fixed_params": {"class_weight": "balanced", "random_state": 42}},
                  f, indent=2)
    return model, search.best_params_, search.best_score_


def evaluate(model, df_test: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    df_test = df_test.copy()
    df_test["proba"] = model.predict_proba(df_test[FEATURES])[:, 1]
    df_test["pred"] = model.predict(df_test[FEATURES])

    rows = []
    for (project, bug), group in df_test.groupby(["project", "bug"]):
        ordered = group.sort_values(["proba", "test_method"], ascending=[False, True])
        labels = ordered["label"].to_numpy()
        rows.append({"project": project, "bug": bug,
                     "apfd": calculate_apfd(labels),
                     "n_tests": len(labels),
                     "n_trigger_tests": int(labels.sum())})
    results = pd.DataFrame(rows).sort_values(["project", "bug"]).reset_index(drop=True)
    results.to_csv(RESULTS_DIR / f"random_forest_apfd{suffix}.csv", index=False)

    print("Precision/Recall/F1 da classificação binária (threshold 0.5) — apenas RF,")
    print("baselines não fazem classificação binária:")
    for name, subset in [("Geral", df_test),
                         ("Lang", df_test[df_test["project"] == "Lang"]),
                         ("Chart", df_test[df_test["project"] == "Chart"])]:
        p, r, f1, _ = precision_recall_fscore_support(
            subset["label"], subset["pred"], average="binary", zero_division=0)
        pos = int(subset["label"].sum())
        print(f"  {name:<6} precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}  "
              f"(instâncias={len(subset)}, positivas={pos})")
    return results


def summary(rf: pd.DataFrame, split: dict):
    test_pairs = {(proj, bug) for proj, bugs in split["test"].items() for bug in bugs}

    def stats(s):
        return f"média={s.mean():.4f}  mediana={s.median():.4f}  dp={s.std(ddof=1):.4f}"

    print(f"\nAPFD do Random Forest ({len(rf)} bugs de teste):")
    print(f"  Geral: {stats(rf['apfd'])}")
    print(rf.groupby("project")["apfd"].agg(["mean", "median", "std"]).round(4).to_string())

    print("\nComparação com as baselines NOS MESMOS bugs de teste:")
    for name, path, col in [("Random", "random_baseline_apfd.csv", "apfd_mean"),
                            ("History", "history_baseline_apfd.csv", "apfd")]:
        base = pd.read_csv(RESULTS_DIR / path)
        base = base[base.apply(lambda r: (r["project"], r["bug"]) in test_pairs, axis=1)]
        print(f"  {name:<8} {stats(base[col])}")
    print(f"  {'RF':<8} {stats(rf['apfd'])}")
    print("\nReferência (todos os 86 bugs válidos): Random=0.5750, History=0.5667.")


def run(scoring: str = "f1"):
    suffix = "" if scoring == "f1" else f"_{scoring.replace('_', '')}"
    df = pd.read_csv(CSV_PATH)
    split = make_split(df)
    print(f"Split por bug salvo em results/train_test_split.json (seed={SPLIT_SEED})")
    for side in ("train", "test"):
        for proj, bugs in split[side].items():
            print(f"  {side} {proj} ({len(bugs)}): {bugs}")

    df_train = df[mask_for(df, split["train"])]
    df_test = df[mask_for(df, split["test"])]
    print(f"\nInstâncias: treino={len(df_train)}, teste={len(df_test)}")

    model, best_params, cv_score = train(df_train, scoring, suffix)
    print(f"\nMelhores hiperparâmetros (GridSearchCV, {scoring}, 5-fold estratificado no treino):")
    print(f"  {best_params}  ({scoring} médio na CV: {cv_score:.4f})")
    print(f"Modelo salvo em results/rf_model{suffix}.joblib, "
          f"hiperparâmetros em results/rf_hyperparameters{suffix}.json\n")

    rf_results = evaluate(model, df_test, suffix)
    summary(rf_results, split)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "f1")
