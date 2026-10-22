"""Boxplots do APFD por bug no conjunto de teste, para as quatro estratégias.

Lê results/apfd_long_format.csv (gerado por src.metrics.consolidate_results);
não recalcula APFD.

Uso: python3 -m src.visualization.plot_apfd
Gera: results/figures/apfd_boxplot_geral.{pdf,png}
      results/figures/apfd_boxplot_por_projeto.{pdf,png}
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
PROJECTS = ("Lang", "Chart", "Math", "Time", "Mockito", "Compress")

# Nome no CSV -> rótulo no texto. A cor acompanha a estratégia em todas as
# figuras; as quatro primeiras cores da paleta categórica validada para
# daltonismo (protanopia, deuteranopia, tritanopia).
STRATEGIES = {
    "Random": "Random",
    "History-based": "History-based",
    "same_package": "Same-Package",
    "Random Forest": "Random Forest",
}
COLORS = {
    "Random": "#2a78d6",
    "History-based": "#eb6834",
    "Same-Package": "#1baf7a",
    "Random Forest": "#eda100",
}
INK = "#333333"
TICK_LABELS = ["Random", "History-\nbased", "Same-\nPackage", "Random\nForest"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,  # fontes TrueType embutidas, editáveis no PDF
})


def boxplot(ax, df: pd.DataFrame, title: str):
    data = [df.loc[df["strategy"] == s, "apfd"] for s in COLORS]
    box = ax.boxplot(
        data, patch_artist=True, showmeans=True, widths=0.6,
        medianprops={"color": INK, "linewidth": 1.5},
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": INK, "markersize": 6},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "none", "markeredgecolor": INK},
        whiskerprops={"color": INK}, capprops={"color": INK},
    )
    for patch, color in zip(box["boxes"], COLORS.values()):
        patch.set(facecolor=color, edgecolor=INK, alpha=0.9)
    ax.set_xticks(range(1, len(COLORS) + 1), TICK_LABELS)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)


def mean_legend(fig):
    handle = Line2D([], [], marker="D", linestyle="none", markerfacecolor="white",
                    markeredgecolor=INK, markersize=6, label="Média")
    median = Line2D([], [], color=INK, linewidth=1.5, label="Mediana")
    fig.legend(handles=[median, handle], loc="outside lower center", frameon=False, ncol=2)


def save(fig, name: str):
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {FIGURES_DIR / name}.pdf / .png")


def run():
    df = pd.read_csv(RESULTS_DIR / "apfd_long_format.csv")
    df["strategy"] = df["strategy"].map(STRATEGIES)
    assert df["strategy"].notna().all(), "estratégia desconhecida em apfd_long_format.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_bugs = lambda d: d.groupby(["project", "bug"]).ngroups  # noqa: E731

    print("Figuras salvas:")
    fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    boxplot(ax, df, f"APFD por bug no conjunto de teste (n = {n_bugs(df)} bugs)")
    ax.set_xlabel("Estratégia")
    ax.set_ylabel("APFD")
    mean_legend(fig)
    save(fig, "apfd_boxplot_geral")

    fig, axes = plt.subplots(3, 2, figsize=(6.0, 8.0), sharey=True, layout="constrained")
    for ax, project in zip(axes.flat, PROJECTS):
        sub = df[df["project"] == project]
        boxplot(ax, sub, f"{project} (n = {n_bugs(sub)} bugs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("APFD")
    for ax in axes[-1]:
        ax.set_xlabel("Estratégia")
    fig.suptitle("APFD por bug no conjunto de teste, por projeto")
    mean_legend(fig)
    save(fig, "apfd_boxplot_por_projeto")


if __name__ == "__main__":
    run()
