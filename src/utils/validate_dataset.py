"""Validação do features.csv."""

from pathlib import Path

import pandas as pd

from config.constants import EXPECTED_BUGS, FEATURE_COLUMNS, PROJECTS

CSV_PATH = "data/processed/features.csv"

PREDICTIVE_FEATURES = ["history", "same_package", "modified_classes_count"]
BUILD_REPORT_PATH = "data/processed/dataset_build_report.csv"


def justified_absences() -> dict:
    """Bugs que o relatório de construção marca como falha, por projeto."""
    path = Path(BUILD_REPORT_PATH)
    if not path.exists():
        return {}

    report = pd.read_csv(path)
    failed = report[report["status"] != "ok"]
    out: dict = {}
    for row in failed.itertuples(index=False):
        out.setdefault(row.project, []).append((int(row.bug), str(row.reason)))
    return out


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

    # Um bug ausente só é aceitável se o relatório de construção registrar o
    # motivo. Sem isso o validador ratificaria a própria falha de coleta.
    justified = justified_absences()

    print("\nBugs por projeto vs. bugs ativos do Defects4J:")
    for project in PROJECTS:
        found = int(bugs_por_projeto.get(project, 0))
        expected = EXPECTED_BUGS[project]
        absent = expected - found
        excused = len(justified.get(project, []))

        if absent == 0:
            status = "OK"
        elif absent == excused:
            status = f"OK ({absent} com falha registrada)"
        else:
            status = "DIVERGE"
            ok = False
            issues.append(
                f"{project}: {found} de {expected} bugs; {absent} ausente(s), "
                f"{excused} com motivo registrado em {BUILD_REPORT_PATH}"
            )
        print(f"  {project:<10} {found:>3} | ativos {expected:>3}  [{status}]")
        if absent and excused:
            for bug_id, reason in justified[project]:
                print(f"             - {project}-{bug_id}: {reason}")

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
        ok = False
        issues.append(
            f"{len(sem_trigger)} bug(s) sem nenhum trigger test — com o enumerador "
            "corrigido isso indica trigger perdido, não bug sem teste que falha"
        )
        print(
            f"\nPROBLEMA — {len(sem_trigger)} bug(s) SEM trigger test (label=1). "
            "O Defects4J garante ao menos um teste que falha por bug, então isso "
            "significa que a enumeração perdeu o trigger:"
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
