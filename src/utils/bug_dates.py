"""Ordem cronológica dos bugs, a partir da data do commit de correção.

Os ids de bug do Defects4J não acompanham o tempo de forma uniforme. Em Lang,
Chart e Math o id cresce para trás (Lang-1 é o LANG-747, de 2011; Lang-65 é o
LANG-59, de 2002), enquanto em Compress ele cresce para frente. Qualquer coisa
que dependa de "bugs anteriores" — a feature `history` e o split cronológico —
precisa da data real, não do id.

As datas vêm de data/processed/bug_commit_dates.csv, gerado por
scripts/export_bug_dates.py a partir dos repositórios git do Defects4J.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from config.constants import DATA_PROCESSED_DIR

DATES_PATH = DATA_PROCESSED_DIR / "bug_commit_dates.csv"


@lru_cache(maxsize=1)
def load_bug_dates(path: str = None) -> Dict[Tuple[str, int], int]:
    """Mapa (project, bug) -> timestamp do commit de correção."""
    dates_path = Path(path) if path else DATES_PATH
    if not dates_path.exists():
        raise FileNotFoundError(
            f"{dates_path} não encontrado. Rode 'python3 scripts/export_bug_dates.py'. "
            "A ordem cronológica não pode cair de volta para a ordem dos ids, que é "
            "invertida em Lang, Chart e Math."
        )
    df = pd.read_csv(dates_path)
    return {
        (row.project, int(row.bug)): int(row.fixed_commit_date)
        for row in df.itertuples(index=False)
    }


def chronological_order(project: str, bugs: Iterable[int]) -> List[int]:
    """Bugs do projeto em ordem cronológica, com o id como desempate."""
    dates = load_bug_dates()
    bugs = [int(b) for b in bugs]

    missing = [b for b in bugs if (project, b) not in dates]
    if missing:
        raise KeyError(
            f"Sem data de commit para {project}: {sorted(missing)}. "
            f"Regenere {DATES_PATH}."
        )

    return sorted(bugs, key=lambda b: (dates[(project, b)], b))


def chronological_rank(df: pd.DataFrame) -> pd.Series:
    """Posição cronológica (0, 1, 2, ...) de cada linha dentro do seu projeto.

    Serve para ordenar um DataFrame de instâncias por tempo sem depender do id.
    """
    dates = load_bug_dates()
    ranks = {}

    for project, group in df.groupby("project"):
        ordered = chronological_order(project, group["bug"].unique())
        ranks.update({(project, bug): i for i, bug in enumerate(ordered)})

    keys = list(zip(df["project"], df["bug"].astype(int)))
    missing = sorted({k for k in keys if k not in ranks})
    if missing:
        raise KeyError(f"Sem ordem cronológica para: {missing}")

    return pd.Series([ranks[k] for k in keys], index=df.index, name="chrono_rank")


if __name__ == "__main__":
    dates = load_bug_dates()
    projects = sorted({p for p, _ in dates})
    for project in projects:
        bugs = [b for p, b in dates if p == project]
        ordered = chronological_order(project, bugs)
        direction = "cresce" if ordered[0] < ordered[-1] else "DECRESCE"
        print(
            f"{project:<9} {len(bugs):>3} bugs | mais antigo {project}-{ordered[0]}"
            f" -> mais recente {project}-{ordered[-1]} | id {direction} com o tempo"
        )
