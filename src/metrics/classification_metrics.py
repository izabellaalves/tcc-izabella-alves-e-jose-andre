"""Precision, Recall, F1 e AUC-ROC do Random Forest no conjunto de teste.

Precision, Recall e F1 só se aplicam ao Random Forest. Random, History-based e
Same-Package são ordenações de testes, não classificadores binários: não têm
limiar de decisão nem predição positiva/negativa, então essas métricas não
existem para elas. A comparação entre as quatro estratégias é pelo APFD.

Reutiliza os modelos salvos e o split de results/train_test_split.json, sem
retreinar. A AUC de validação cruzada refaz só os folds (GroupKFold por
project+bug, o mesmo esquema do GridSearchCV) com os hiperparâmetros de cada
modelo; os modelos salvos não mudam.

Uso: python3 -m src.metrics.classification_metrics
Gera: results/classification_metrics.csv, results/auc_metrics.csv
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score

from src.ml.random_forest_pipeline import CSV_PATH, FEATURES, mask_for

RESULTS_DIR = Path("results")
PROJECTS = ("Lang", "Chart", "Math", "Time", "Mockito", "Compress")

# modelo -> sufixo dos arquivos gerados por src.ml.random_forest_pipeline
MODELS = {
    "Random Forest (F1, oficial)": "",
    "Random Forest (AUC-ROC, alternativo)": "_rocauc",
}


def load_model(suffix: str):
    model = joblib.load(RESULTS_DIR / f"rf_model{suffix}.joblib")
    with open(RESULTS_DIR / f"rf_hyperparameters{suffix}.json") as f:
        best = json.load(f)["best_params"]
    params = model.get_params()
    if any(params[k] != v for k, v in best.items()):
        raise SystemExit(f"rf_model{suffix}.joblib não corresponde a rf_hyperparameters{suffix}.json; "
                         "rode o pipeline do Random Forest de novo.")
    return model


def classification_rows(model_name: str, df_test: pd.DataFrame) -> list:
    rows = []
    for scope, subset in [("Geral", df_test)] + [(p, df_test[df_test["project"] == p]) for p in PROJECTS]:
        tn, fp, fn, tp = confusion_matrix(subset["label"], subset["pred"], labels=[0, 1]).ravel()
        p, r, f1, _ = precision_recall_fscore_support(
            subset["label"], subset["pred"], average="binary", zero_division=0)
        rows.append({"modelo": model_name, "escopo": scope,
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                     "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                     "n_positivos": int(subset["label"].sum()), "n_instancias": len(subset),
                     "n_bugs": subset.groupby(["project", "bug"]).ngroups})
    # Os projetos particionam o teste: as contagens por projeto somam as do Geral.
    geral, *por_projeto = rows
    for k in ("tp", "fp", "fn", "tn", "n_positivos", "n_instancias", "n_bugs"):
        assert geral[k] == sum(r[k] for r in por_projeto), k
    return rows


def run():
    df = pd.read_csv(CSV_PATH)
    with open(RESULTS_DIR / "train_test_split.json") as f:
        split = json.load(f)
    df_train = df[mask_for(df, split["train"])]
    df_test = df[mask_for(df, split["test"])].copy()
    groups = df_train["project"] + "-" + df_train["bug"].astype(str)

    metric_rows, auc_rows = [], []
    for name, suffix in MODELS.items():
        model = load_model(suffix)
        df_test["pred"] = model.predict(df_test[FEATURES])  # limiar padrão de 0,5
        proba = model.predict_proba(df_test[FEATURES])[:, 1]
        metric_rows += classification_rows(name, df_test)

        cv = cross_val_score(clone(model), df_train[FEATURES], df_train["label"], groups=groups,
                             cv=GroupKFold(n_splits=5), scoring="roc_auc")
        auc_rows.append({"modelo": name, "min_samples_leaf": model.min_samples_leaf,
                         "cv_auc_mean": round(cv.mean(), 4), "cv_auc_std": round(cv.std(), 4),
                         "cv": "GroupKFold(5) por project+bug, no treino",
                         "test_auc": round(roc_auc_score(df_test["label"], proba), 4),
                         "test": "por instância, 89 bugs de teste"})

    metrics = pd.DataFrame(metric_rows)
    aucs = pd.DataFrame(auc_rows)
    metrics.to_csv(RESULTS_DIR / "classification_metrics.csv", index=False)
    aucs.to_csv(RESULTS_DIR / "auc_metrics.csv", index=False)

    print("Precision/Recall/F1 da classe positiva (trigger), predict com limiar 0,5.")
    print("Aplicam-se só ao Random Forest: Random, History-based e Same-Package são")
    print("ordenações, não classificadores binários.\n")
    print(metrics.to_string(index=False))
    print("\nAUC-ROC:")
    print(aucs.to_string(index=False))
    print("\nSalvo em results/classification_metrics.csv e results/auc_metrics.csv")


if __name__ == "__main__":
    run()
