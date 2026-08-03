"""Cálculo de features para cada par (bug, método de teste)."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd

from src.defects4j.metadata_exporter import BugMetadata
from src.feature_engineering.test_enumerator import TestMethod
from src.utils.helpers import extract_package_name
from src.utils.logger import get_logger


@dataclass
class TestMethodRow:
    """Linha da base intermediária."""

    project: str
    bug: int
    test_class: str
    test_method: str
    is_trigger: int
    modified_classes: str
    modified_classes_count: int


@dataclass
class FeatureRow:
    """Linha da base final com features."""

    project: str
    bug: int
    test_class: str
    test_method: str
    history: int
    same_package: int
    modified_classes_count: int
    label: int


class FeatureEngineer:
    """Calcula features para treinamento do Random Forest."""

    def __init__(self):
        self.logger = get_logger()

    def build_intermediate_table(
        self,
        metadatas: List[BugMetadata],
        test_methods_dict: Dict[str, List[TestMethod]],
    ) -> pd.DataFrame:
        """Constrói tabela com uma linha por par (bug, método de teste)."""
        self.logger.info("Construindo tabela intermediária")

        rows = []
        sorted_metadatas = sorted(metadatas, key=lambda m: (m.project, m.bug_id))

        for metadata in sorted_metadatas:
            bug_name = str(metadata)
            test_methods = test_methods_dict.get(bug_name, [])

            if not test_methods:
                self.logger.warning(
                    "%s: nenhum método de teste encontrado, pulando",
                    bug_name,
                )
                continue

            modified_classes_str = ";".join(metadata.modified_classes)

            for test_method in test_methods:
                trigger_signature = str(test_method)
                is_trigger = 1 if trigger_signature in metadata.trigger_tests else 0

                rows.append(
                    asdict(
                        TestMethodRow(
                            project=metadata.project,
                            bug=metadata.bug_id,
                            test_class=test_method.class_name,
                            test_method=test_method.method_name,
                            is_trigger=is_trigger,
                            modified_classes=modified_classes_str,
                            modified_classes_count=metadata.modified_classes_count,
                        )
                    )
                )

        df = pd.DataFrame(rows)

        self.logger.info(
            "Tabela intermediária: %d linhas, %d projetos, %d bugs",
            len(df),
            df["project"].nunique(),
            df["bug"].nunique(),
        )

        trigger_count = df["is_trigger"].sum()
        non_trigger_count = len(df) - trigger_count
        self.logger.info(
            "Triggers: %d (%.2f%%)",
            trigger_count,
            trigger_count / len(df) * 100,
        )
        self.logger.info(
            "Non-triggers: %d (%.2f%%)",
            non_trigger_count,
            non_trigger_count / len(df) * 100,
        )

        return df

    def calculate_history_feature(self, df: pd.DataFrame) -> pd.Series:
        """Calcula history com base apenas em bugs anteriores do mesmo projeto."""
        self.logger.info("Calculando feature 'history'")

        df = df.sort_values(
            ["project", "bug", "test_class", "test_method"]
        ).reset_index(drop=True)

        history_values = []
        project_histories = defaultdict(lambda: defaultdict(list))

        current_project = None
        current_bug = None

        for _, row in df.iterrows():
            project = row["project"]
            bug_id = row["bug"]
            key = (row["test_class"], row["test_method"])

            if current_project != project or current_bug != bug_id:
                current_project = project
                current_bug = bug_id

            history_values.append(len(project_histories[project][key]))

            if row["is_trigger"] == 1:
                project_histories[project][key].append(bug_id)

        history_series = pd.Series(history_values, index=df.index)

        self.logger.info(
            "History: min=%d, max=%d, mean=%.2f",
            history_series.min(),
            history_series.max(),
            history_series.mean(),
        )

        return history_series

    def calculate_same_package_feature(
        self,
        test_class: str,
        modified_classes_str: str,
    ) -> int:
        """Retorna 1 se a classe de teste está no mesmo pacote de alguma classe modificada."""
        if not modified_classes_str:
            return 0

        test_package = extract_package_name(test_class)
        modified_packages = {
            extract_package_name(cls)
            for cls in modified_classes_str.split(";")
            if cls.strip()
        }

        return 1 if test_package in modified_packages else 0

    def calculate_features(self, df_intermediate: pd.DataFrame) -> pd.DataFrame:
        """Calcula todas as features a partir da tabela intermediária."""
        self.logger.info("Calculando features")

        df = df_intermediate.sort_values(["project", "bug"]).copy()
        df["history"] = self.calculate_history_feature(df)

        self.logger.info("Calculando feature 'same_package'")
        df["same_package"] = df.apply(
            lambda row: self.calculate_same_package_feature(
                row["test_class"],
                row["modified_classes"],
            ),
            axis=1,
        )

        same_package_pct = df["same_package"].sum() / len(df) * 100
        self.logger.info(
            "same_package=1: %d (%.2f%%)",
            df["same_package"].sum(),
            same_package_pct,
        )

        df["label"] = df["is_trigger"]

        df_features = df[
            [
                "project",
                "bug",
                "test_class",
                "test_method",
                "history",
                "same_package",
                "modified_classes_count",
                "label",
            ]
        ].copy()

        self.logger.info("Features calculadas: %d linhas", len(df_features))

        return df_features

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Valida colunas, tipos e distribuição das features."""
        self.logger.info("Validando features")

        errors = []
        required_cols = [
            "project",
            "bug",
            "test_class",
            "test_method",
            "history",
            "same_package",
            "modified_classes_count",
            "label",
        ]

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Colunas faltando: {missing_cols}")

        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            errors.append(f"Valores nulos encontrados:\n{null_counts[null_counts > 0]}")

        if df["history"].dtype not in ["int64", "int32"]:
            errors.append(f"history deve ser int, encontrado: {df['history'].dtype}")

        if not df["same_package"].isin([0, 1]).all():
            errors.append("same_package deve conter apenas 0 ou 1")

        if not df["label"].isin([0, 1]).all():
            errors.append("label deve conter apenas 0 ou 1")

        if (df["history"] < 0).any():
            errors.append("history não pode ser negativo")

        if (df["modified_classes_count"] < 0).any():
            errors.append("modified_classes_count não pode ser negativo")

        if len(df["label"].value_counts()) < 2:
            errors.append("label deve ter pelo menos 2 classes distintas")

        if errors:
            self.logger.error("Validação falhou:")
            for error in errors:
                self.logger.error("  - %s", error)
            return False

        self.logger.info("Validação concluída com sucesso")
        return True

    def print_statistics(self, df: pd.DataFrame):
        """Imprime estatísticas descritivas das features."""
        self.logger.info("Estatísticas das features")
        self.logger.info("Total de linhas: %d", len(df))
        self.logger.info("Projetos: %s", df["project"].unique().tolist())
        self.logger.info("Bugs únicos: %d", df["bug"].nunique())
        self.logger.info("Classes de teste únicas: %d", df["test_class"].nunique())

        self.logger.info("Distribuição de labels:")
        for label, count in df["label"].value_counts().sort_index().items():
            pct = count / len(df) * 100
            self.logger.info("  label=%s: %d (%.2f%%)", label, count, pct)

        self.logger.info("Feature 'history':")
        self.logger.info("  Min: %s", df["history"].min())
        self.logger.info("  Max: %s", df["history"].max())
        self.logger.info("  Mean: %.2f", df["history"].mean())
        self.logger.info("  Median: %.1f", df["history"].median())

        self.logger.info("Feature 'same_package':")
        for val, count in df["same_package"].value_counts().sort_index().items():
            pct = count / len(df) * 100
            self.logger.info("  %s: %d (%.2f%%)", val, count, pct)

        self.logger.info("Feature 'modified_classes_count':")
        self.logger.info("  Min: %s", df["modified_classes_count"].min())
        self.logger.info("  Max: %s", df["modified_classes_count"].max())
        self.logger.info("  Mean: %.2f", df["modified_classes_count"].mean())
        self.logger.info("  Median: %.1f", df["modified_classes_count"].median())
