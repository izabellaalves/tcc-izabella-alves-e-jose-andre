#!/usr/bin/env python3
"""Exporta a data de commit do fix de cada bug, para ordenar o split cronologicamente.

Os ids de bug do Defects4J NÃO seguem a ordem do tempo de forma uniforme: em
Lang, Chart e Math o id cresce para trás (Lang-1 é o LANG-747, de 2011, e
Lang-65 é o LANG-59, de 2002), enquanto em Compress ele cresce para frente.
Ordenar o split por id, portanto, treina no futuro e testa no passado em parte
dos projetos. Este script resolve isso lendo a data real do commit de correção
(`revision.id.fixed` de active-bugs.csv) no repositório git do projeto.

Uso: python3 scripts/export_bug_dates.py
Gera: data/processed/bug_commit_dates.csv (project, bug, fixed_commit_date, fixed_revision)

O CSV é versionado para que o pipeline de ML rode sem precisar dos repositórios
clonados (defects4j/project_repos/, que não vão para o git).
"""

import csv
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import DATA_PROCESSED_DIR, DEFECTS4J_DIR, PROJECTS

# Nome do repositório e sistema de controle de versão, conforme
# framework/core/Project/<P>.pm. Chart é o único em SVN (Vcs::Svn).
REPOS = {
    "Lang": ("commons-lang.git", "git"),
    "Chart": ("jfreechart", "svn"),
    "Math": ("commons-math.git", "git"),
    "Time": ("joda-time.git", "git"),
    "Mockito": ("mockito.git", "git"),
    "Compress": ("commons-compress.git", "git"),
}

OUTPUT_PATH = DATA_PROCESSED_DIR / "bug_commit_dates.csv"


def git_timestamp(repo: Path, revision: str) -> Optional[int]:
    """Timestamp do committer da revisão, ou None se a revisão não existe."""
    result = subprocess.run(
        ["git", "--git-dir", str(repo), "show", "-s", "--format=%ct", revision],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()
    return int(line[-1]) if line and line[-1].isdigit() else None


def svn_timestamp(repo: Path, revision: str) -> Optional[int]:
    """Timestamp de uma revisão SVN, lido direto do revprop do repositório FSFS.

    Evita depender do cliente svn, que não é requisito do Defects4J. O arquivo
    db/revprops/<shard>/<rev> guarda svn:date no formato ISO com microssegundos.
    """
    if not revision.isdigit():
        return None
    rev = int(revision)

    shard_size = 1000
    layout = (repo / "db" / "format").read_text(errors="replace")
    match = re.search(r"layout sharded (\d+)", layout)
    if match:
        shard_size = int(match.group(1))

    candidates = [
        repo / "db" / "revprops" / str(rev // shard_size) / str(rev),
        repo / "db" / "revprops" / str(rev),
    ]
    revprop = next((c for c in candidates if c.exists()), None)
    if revprop is None:
        return None

    content = revprop.read_text(errors="replace")
    match = re.search(r"svn:date\nV \d+\n(\S+)", content)
    if not match:
        return None

    stamp = match.group(1).replace("Z", "+00:00")
    return int(datetime.fromisoformat(stamp).astimezone(timezone.utc).timestamp())


def commit_timestamp(repo: Path, revision: str, vcs: str) -> Optional[int]:
    return git_timestamp(repo, revision) if vcs == "git" else svn_timestamp(repo, revision)


def active_bugs(project: str) -> List[Tuple[int, str]]:
    path = DEFECTS4J_DIR / "framework" / "projects" / project / "active-bugs.csv"
    with open(path, encoding="utf-8") as f:
        return [
            (int(row["bug.id"]), row["revision.id.fixed"])
            for row in csv.DictReader(f)
        ]


def main() -> int:
    repo_dir = DEFECTS4J_DIR / "project_repos"
    rows = []
    failures = []

    for project in PROJECTS:
        repo_name, vcs = REPOS[project]
        repo = repo_dir / repo_name
        if not repo.exists():
            print(f"ERRO: repositório não encontrado: {repo}")
            print("Rode defects4j/init.sh (ou project_repos/get_repos.sh) primeiro.")
            return 1

        bugs = active_bugs(project)
        for bug_id, revision in bugs:
            ts = commit_timestamp(repo, revision, vcs)
            if ts is None:
                failures.append((project, bug_id, revision))
                continue
            rows.append(
                {
                    "project": project,
                    "bug": bug_id,
                    "fixed_commit_date": ts,
                    "fixed_revision": revision,
                }
            )

        got = [r for r in rows if r["project"] == project]
        ordered = sorted(got, key=lambda r: r["fixed_commit_date"])
        if ordered:
            first, last = ordered[0], ordered[-1]
            direction = "id CRESCE com o tempo" if first["bug"] < last["bug"] else "id DECRESCE com o tempo"
            print(
                f"{project:<9} {len(got):>3} bugs | mais antigo: {project}-{first['bug']:<4}"
                f" mais recente: {project}-{last['bug']:<4} | {direction}"
            )

    if failures:
        print("\nRevisões não encontradas no repositório:")
        for project, bug_id, revision in failures:
            print(f"  {project}-{bug_id}: {revision}")

    if not rows:
        print("Nenhuma data exportada.")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["project", "bug", "fixed_commit_date", "fixed_revision"]
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["project"], r["bug"])))

    print(f"\nSalvo: {OUTPUT_PATH} ({len(rows)} bugs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
