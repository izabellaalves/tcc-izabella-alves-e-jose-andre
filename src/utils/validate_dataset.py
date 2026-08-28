"""Validação cruzada do features.csv com os metadados da Tabela 6 (capítulo 4).

Uso: python3 -m src.utils.validate_dataset
"""

import pandas as pd

CSV_PATH = "data/processed/features.csv"

EXPECTED = {
    "projetos": 2,
    "bugs": 87,
    "bugs Lang": 61,
    "bugs Chart": 26,
    "instâncias": 16376,
    "atributos": 8,
    "features preditivas": 3,
    "instâncias positivas (label=1)": 208,
    "instâncias negativas (label=0)": 16168,
}

PREDICTIVE_FEATURES = ["history", "same_package", "modified_classes_count"]


def validate(csv_path: str = CSV_PATH) -> bool:
    df = pd.read_csv(csv_path)

    print(f"Arquivo: {csv_path}")
    print(f"Linhas (instâncias): {len(df)}")
    print(f"Colunas ({len(df.columns)}): {list(df.columns)}")
    print("\nTipos de dados:")
    print(df.dtypes.to_string())

    bugs = df.groupby(["project", "bug"]).ngroups
    bugs_por_projeto = df.groupby("project")["bug"].nunique()
    print(f"\nBugs únicos (project+bug): {bugs}")
    print("Bugs por projeto:")
    print(bugs_por_projeto.to_string())

    label_counts = df["label"].value_counts()
    print("\nDistribuição de label:")
    print(label_counts.to_string())

    nulls = df.isnull().sum()
    if nulls.any():
        print("\nATENÇÃO — valores nulos encontrados:")
        print(nulls[nulls > 0].to_string())
    else:
        print("\nValores nulos: nenhum.")

    triggers_por_bug = df.groupby(["project", "bug"])["label"].sum()
    sem_trigger = triggers_por_bug[triggers_por_bug == 0]
    if len(sem_trigger) > 0:
        print(f"\nPROBLEMA — {len(sem_trigger)} bug(s) SEM trigger test (label=1), "
              "o APFD não poderá ser calculado para eles:")
        print(sem_trigger.to_string())
    else:
        print("Bugs sem trigger test: nenhum (todos os bugs têm ao menos um label=1).")

    found = {
        "projetos": df["project"].nunique(),
        "bugs": bugs,
        "bugs Lang": int(bugs_por_projeto.get("Lang", 0)),
        "bugs Chart": int(bugs_por_projeto.get("Chart", 0)),
        "instâncias": len(df),
        "atributos": len(df.columns),
        "features preditivas": sum(c in df.columns for c in PREDICTIVE_FEATURES),
        "instâncias positivas (label=1)": int(label_counts.get(1, 0)),
        "instâncias negativas (label=0)": int(label_counts.get(0, 0)),
    }

    print("\n" + "=" * 62)
    print("RESUMO — encontrado vs. esperado (Tabela 6)")
    print("=" * 62)
    ok = True
    for key, expected in EXPECTED.items():
        got = found[key]
        status = "OK" if got == expected else "DIVERGE"
        ok = ok and got == expected
        print(f"{key:<34} {got:>8} | esperado {expected:>6}  [{status}]")
    print("=" * 62)
    print("RESULTADO:", "dataset CONSISTENTE com a Tabela 6."
          if ok else "há DIVERGÊNCIAS — revisar antes de prosseguir.")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
