"""Validação do features.csv."""

import pandas as pd

from config.constants import EXPECTED_BUGS, FEATURE_COLUMNS, PROJECTS

CSV_PATH = "data/processed/features.csv"

PREDICTIVE_FEATURES = ["history", "same_package", "modified_classes_count"]
PROJECTS_WITH_OPTIONAL_EXTRA_BUGS = {"Lang", "Math"}


def validate(csv_path: str = CSV_PATH) -> bool:
    df = pd.read_csv(csv_path)

    print(f"Arquivo: {csv_path}")
    print(f"Linhas (instâncias): {len(df)}")
    print(f"Colunas ({len(df.columns)}): {list(df.columns)}")
    print("\nTipos de dados:")
    print(df.dtypes.to_string())

    ok = True
    issues = []

    projects_found = sorted(df["project"].unique())
    print(f"\nProjetos encontrados: {projects_found}")
    if projects_found != sorted(PROJECTS):
        ok = False
        issues.append(f"projetos esperados {sorted(PROJECTS)}, encontrados {projects_found}")

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in FEATURE_COLUMNS]
    if missing_cols:
        ok = False
        issues.append(f"colunas faltando: {missing_cols}")
    if extra_cols:
        ok = False
        issues.append(f"colunas extras: {extra_cols}")

    missing_features = [c for c in PREDICTIVE_FEATURES if c not in df.columns]
    if missing_features:
        ok = False
        issues.append(f"features preditivas faltando: {missing_features}")

    bugs = df.groupby(["project", "bug"]).ngroups
    bugs_por_projeto = df.groupby("project")["bug"].nunique()
    print(f"\nBugs únicos (project+bug): {bugs}")
    print("Bugs por projeto:")
    print(bugs_por_projeto.to_string())

    print("\nBugs por projeto vs. esperado:")
    for project in PROJECTS:
        found = int(bugs_por_projeto.get(project, 0))
        expected = EXPECTED_BUGS[project]
        if project in PROJECTS_WITH_OPTIONAL_EXTRA_BUGS:
            status = "OK" if found >= expected else "ABAIXO"
            if found < expected:
                ok = False
                issues.append(f"{project}: {found} bugs (mínimo esperado {expected})")
        else:
            status = "OK" if found == expected else "DIVERGE"
            if found != expected:
                ok = False
                issues.append(f"{project}: {found} bugs (esperado {expected})")
        print(f"  {project:<10} {found:>3} | esperado {expected:>3}  [{status}]")

    label_counts = df["label"].value_counts()
    print("\nDistribuição de label:")
    print(label_counts.to_string())
    if label_counts.get(1, 0) == 0:
        ok = False
        issues.append("nenhuma instância com label=1")

    nulls = df.isnull().sum()
    if nulls.any():
        ok = False
        issues.append("valores nulos encontrados")
        print("\nATENÇÃO — valores nulos encontrados:")
        print(nulls[nulls > 0].to_string())
    else:
        print("\nValores nulos: nenhum.")

    key_cols = ["project", "bug", "test_class", "test_method"]
    duplicates = df.duplicated(subset=key_cols).sum()
    if duplicates > 0:
        ok = False
        issues.append(f"{duplicates} linhas duplicadas")
        print(f"\nPROBLEMA — {duplicates} linha(s) duplicada(s) em {key_cols}")
    else:
        print("\nDuplicatas: nenhuma.")

    triggers_por_bug = df.groupby(["project", "bug"])["label"].sum()
    sem_trigger = triggers_por_bug[triggers_por_bug == 0]
    if len(sem_trigger) > 0:
        print(
            f"\nAVISO — {len(sem_trigger)} bug(s) SEM trigger test (label=1). "
            "Eles devem ficar no treino, não no teste:"
        )
        print(sem_trigger.to_string())
    else:
        print("\nBugs sem trigger test: nenhum.")

    print("\n" + "=" * 62)
    print("RESUMO")
    print("=" * 62)
    print(f"{'projetos':<34} {len(projects_found):>8}")
    print(f"{'bugs':<34} {bugs:>8}")
    print(f"{'instâncias':<34} {len(df):>8}")
    print(f"{'atributos':<34} {len(df.columns):>8}")
    print(f"{'features preditivas':<34} {len(PREDICTIVE_FEATURES):>8}")
    print(f"{'instâncias positivas (label=1)':<34} {int(label_counts.get(1, 0)):>8}")
    print(f"{'instâncias negativas (label=0)':<34} {int(label_counts.get(0, 0)):>8}")
    print(f"{'bugs sem trigger':<34} {len(sem_trigger):>8}")
    print("=" * 62)

    if issues:
        print("Problemas encontrados:")
        for issue in issues:
            print(f"  - {issue}")

    if ok:
        print("RESULTADO: dataset válido para os experimentos.")
    else:
        print("RESULTADO: há problemas estruturais — revisar antes de prosseguir.")

    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
