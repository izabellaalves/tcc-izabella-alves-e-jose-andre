"""Pipeline Random Forest: split por bug, treino com tuning e avaliação (APFD + P/R/F1)."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.metrics.apfd import calculate_apfd

CSV_PATH = "data/processed/features.csv"
RESULTS_DIR = Path("results")

TEST_RATIO = 0.30
FEATURES = [
    "history",
    "same_package",
    "modified_classes_count",
]


def make_split(df: pd.DataFrame) -> dict:
    split = {"method": "chronological_70_30", "train": {}, "test": {}}
    projects = sorted(df["project"].unique())

    for project in projects:
        df_project = df[df["project"] == project]
        all_bugs = sorted(df_project["bug"].unique().tolist())
        bugs_with_no_trigger = []
        bugs_with_trigger = []

        for bug_id in all_bugs:
            df_bug = df_project[df_project["bug"] == bug_id]
            if df_bug["label"].sum() == 0:
                bugs_with_no_trigger.append(int(bug_id))
            else:
                bugs_with_trigger.append(int(bug_id))

        n_test = round(len(bugs_with_trigger) * TEST_RATIO)
        test_bugs = bugs_with_trigger[-n_test:] if n_test > 0 else []
        train_bugs_eligible = bugs_with_trigger[:-n_test] if n_test > 0 else bugs_with_trigger

        split["train"][project] = sorted(bugs_with_no_trigger + train_bugs_eligible)
        split["test"][project] = sorted(test_bugs)

    summary = {
        "split_method": "chronological_70_30_by_bug_id",
        "test_ratio": TEST_RATIO,
        "projects": {},
    }

    for project in projects:
        df_project = df[df["project"] == project]
        no_trigger = sum(
            1 for b in split["train"][project]
            if df_project[df_project["bug"] == b]["label"].sum() == 0
        )
        summary["projects"][project] = {
            "total_bugs": df_project["bug"].nunique(),
            "bugs_without_trigger": no_trigger,
            "train": len(split["train"][project]),
            "test": len(split["test"][project]),
        }

    split["summary"] = summary

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
    groups = df_train["bug"]
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "min_samples_split": [2, 5, 10],
    }
    search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid,
        scoring=scoring,
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    model = search.best_estimator_
    joblib.dump(model, RESULTS_DIR / f"rf_model{suffix}.joblib")
    with open(RESULTS_DIR / f"rf_hyperparameters{suffix}.json", "w") as f:
        json.dump({"best_params": search.best_params_,
                   f"cv_best_{scoring}": search.best_score_,
                   "scoring": scoring, "cv": "GroupKFold(5, grouped by bug)",
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

    bugs_sem_trigger_no_teste = rf[rf["n_trigger_tests"] == 0]
    if not bugs_sem_trigger_no_teste.empty:
        print("\nAVISO: Bugs sem trigger encontrados no conjunto de teste:")
        print(bugs_sem_trigger_no_teste[["project", "bug", "n_tests"]].to_string(index=False))
    else:
        print("\nOK: Todos os bugs de teste tem pelo menos 1 trigger test")

    print("\nComparação com as baselines NOS MESMOS bugs de teste:")
    for name, path, col in [("Random", "random_baseline_apfd.csv", "apfd_mean"),
                            ("History", "history_baseline_apfd.csv", "apfd")]:
        base = pd.read_csv(RESULTS_DIR / path)
        base = base[base.apply(lambda r: (r["project"], r["bug"]) in test_pairs, axis=1)]
        print(f"  {name:<8} {stats(base[col])}")
    print(f"  {'RF':<8} {stats(rf['apfd'])}")


def run(scoring: str = "f1"):
    suffix = "" if scoring == "f1" else f"_{scoring.replace('_', '')}"
    df = pd.read_csv(CSV_PATH)
    split = make_split(df)

    print(f"\n{'='*80}")
    print("Split crescente 70/30 por id de bug criado")
    print(f"{'='*80}")
    print("Metodo: ultimos 30% bugs com trigger -> teste, primeiros 70% + sem trigger -> treino")
    print("\nResumo por projeto:")
    for proj in sorted(split["test"].keys()):
        n_train = len(split["train"][proj])
        n_test = len(split["test"][proj])
        print(f"  {proj:<10} treino: {n_train:2d}  teste: {n_test:2d}  total: {n_train + n_test:2d}")
        if "summary" in split and proj in split["summary"]["projects"]:
            no_trigger = split["summary"]["projects"][proj]["bugs_without_trigger"]
            if no_trigger > 0:
                print(f"             (inclui {no_trigger} bug(s) sem trigger no treino)")

    print("\nSplit salvo em results/train_test_split.json")
    print(f"{'='*80}\n")

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
