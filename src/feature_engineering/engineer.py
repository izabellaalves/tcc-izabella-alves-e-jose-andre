"""Cálculo de features para cada par (bug, método de teste)."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd

from src.defects4j.metadata_exporter import BugMetadata
from src.feature_engineering.test_enumerator import TestMethod
from src.utils.environment import EnvironmentConfig
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
    historical_failure_rate: float
    last_failure_distance: int
    test_name_similarity: float
    label: int


class FeatureEngineer:
    """Calcula features para treinamento do Random Forest."""

    def __init__(self, config: EnvironmentConfig):
        self.logger = get_logger()
        self.config = config

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

        df_sorted = df.sort_values(
            ["project", "bug", "test_class", "test_method"]
        ).copy()

        history_dict = {}
        project_histories = defaultdict(lambda: defaultdict(list))

        current_project = None
        current_bug = None

        for idx, row in df_sorted.iterrows():
            project = row["project"]
            bug_id = row["bug"]
            key = (row["test_class"], row["test_method"])

            if current_project != project or current_bug != bug_id:
                current_project = project
                current_bug = bug_id

            history_dict[idx] = len(project_histories[project][key])

            if row["is_trigger"] == 1:
                project_histories[project][key].append(bug_id)

        history_series = pd.Series(history_dict)

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

    def calculate_features(
        self,
        df_intermediate: pd.DataFrame,
        metadatas: List[BugMetadata],
        test_methods_dict: Dict[str, List[TestMethod]],
    ) -> pd.DataFrame:
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

        self.logger.info("Calculando feature 'historical_failure_rate'")
        df["historical_failure_rate"] = self.calculate_historical_failure_rate(df)

        self.logger.info("Calculando feature 'last_failure_distance'")
        df["last_failure_distance"] = self.calculate_last_failure_distance(df)

        self.logger.info("Calculando feature 'test_name_similarity'")
        df["test_name_similarity"] = df.apply(
            lambda row: self.calculate_test_name_similarity(
                row["test_class"],
                row["modified_classes"],
            ),
            axis=1,
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
                "historical_failure_rate",
                "last_failure_distance",
                "test_name_similarity",
                "label",
            ]
        ].copy()

        self.logger.info("Features calculadas: %d linhas", len(df_features))

        return df_features

    def calculate_historical_failure_rate(self, df: pd.DataFrame) -> pd.Series:
        self.logger.info("Calculando historical_failure_rate")

        df_sorted = df.sort_values(
            ["project", "bug", "test_class", "test_method"]
        ).copy()

        failure_rate_dict = {}
        project_bug_triggers = defaultdict(lambda: defaultdict(list))

        current_project = None
        current_bug = None

        for idx, row in df_sorted.iterrows():
            project = row["project"]
            bug_id = row["bug"]
            key = (row["test_class"], row["test_method"])

            if current_project != project or current_bug != bug_id:
                current_project = project
                current_bug = bug_id

            triggers_in_past = project_bug_triggers[project][key]
            total_bugs_before = bug_id - 1 if project == "Lang" else bug_id - 1
            
            if total_bugs_before <= 0:
                failure_rate_dict[idx] = 0.0
            else:
                num_failures = len(triggers_in_past)
                failure_rate = num_failures / total_bugs_before
                failure_rate_dict[idx] = round(failure_rate, 4)

            if row["is_trigger"] == 1:
                project_bug_triggers[project][key].append(bug_id)

        failure_rate_series = pd.Series(failure_rate_dict)

        self.logger.info(
            "historical_failure_rate: min=%.4f, max=%.4f, mean=%.4f",
            failure_rate_series.min(),
            failure_rate_series.max(),
            failure_rate_series.mean(),
        )

        return failure_rate_series

    def calculate_last_failure_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calcula a distância desde a última falha do teste.
        
        Retorna quantos bugs passaram desde a última vez que o teste falhou.
        Se o teste nunca falhou antes, retorna o bug_id atual (máxima distância).
        """
        self.logger.info("Calculando last_failure_distance")

        df_sorted = df.sort_values(
            ["project", "bug", "test_class", "test_method"]
        ).copy()

        distance_dict = {}
        project_last_failure = defaultdict(lambda: defaultdict(lambda: None))

        current_project = None
        current_bug = None

        for idx, row in df_sorted.iterrows():
            project = row["project"]
            bug_id = row["bug"]
            key = (row["test_class"], row["test_method"])

            if current_project != project or current_bug != bug_id:
                current_project = project
                current_bug = bug_id

            last_fail_bug = project_last_failure[project][key]
            
            if last_fail_bug is None:
                distance_dict[idx] = bug_id
            else:
                distance_dict[idx] = bug_id - last_fail_bug

            if row["is_trigger"] == 1:
                project_last_failure[project][key] = bug_id

        distance_series = pd.Series(distance_dict)

        self.logger.info(
            "last_failure_distance: min=%d, max=%d, mean=%.2f",
            distance_series.min(),
            distance_series.max(),
            distance_series.mean(),
        )

        return distance_series

    def calculate_test_name_similarity(
        self,
        test_class: str,
        modified_classes_str: str,
    ) -> float:
        """Calcula similaridade entre nome do teste e classes modificadas.
        
        Usa similaridade de Jaccard baseada em tokens do nome.
        Retorna o máximo de similaridade entre o teste e qualquer classe modificada.
        """
        if not modified_classes_str:
            return 0.0

        test_name = test_class.split(".")[-1].lower()
        test_tokens = set(self._tokenize_name(test_name))

        if not test_tokens:
            return 0.0

        max_similarity = 0.0
        modified_classes = [c.strip() for c in modified_classes_str.split(";") if c.strip()]

        for modified_class in modified_classes:
            class_name = modified_class.split(".")[-1].lower()
            class_tokens = set(self._tokenize_name(class_name))

            if not class_tokens:
                continue

            intersection = len(test_tokens & class_tokens)
            union = len(test_tokens | class_tokens)

            if union > 0:
                similarity = intersection / union
                max_similarity = max(max_similarity, similarity)

        return round(max_similarity, 4)

    def _tokenize_name(self, name: str) -> list:
        """Tokeniza um nome de classe em palavras (CamelCase e underscore)."""
        import re
        name = name.replace("test", "").replace("Test", "")
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
        return [t.lower() for t in tokens if len(t) > 1]

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
            "historical_failure_rate",
            "last_failure_distance",
            "test_name_similarity",
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

        if (df["historical_failure_rate"] < 0).any() or (df["historical_failure_rate"] > 1).any():
            errors.append("historical_failure_rate deve estar entre 0 e 1")

        if (df["last_failure_distance"] < 0).any():
            errors.append("last_failure_distance não pode ser negativo")

        if (df["test_name_similarity"] < 0).any() or (df["test_name_similarity"] > 1).any():
            errors.append("test_name_similarity deve estar entre 0 e 1")

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

        self.logger.info("Feature 'historical_failure_rate':")
        self.logger.info("  Min: %.4f", df["historical_failure_rate"].min())
        self.logger.info("  Max: %.4f", df["historical_failure_rate"].max())
        self.logger.info("  Mean: %.4f", df["historical_failure_rate"].mean())
        self.logger.info("  Median: %.4f", df["historical_failure_rate"].median())

        self.logger.info("Feature 'last_failure_distance':")
        self.logger.info("  Min: %d", df["last_failure_distance"].min())
        self.logger.info("  Max: %d", df["last_failure_distance"].max())
        self.logger.info("  Mean: %.2f", df["last_failure_distance"].mean())
        self.logger.info("  Median: %.1f", df["last_failure_distance"].median())

        self.logger.info("Feature 'test_name_similarity':")
        self.logger.info("  Min: %.4f", df["test_name_similarity"].min())
        self.logger.info("  Max: %.4f", df["test_name_similarity"].max())
        self.logger.info("  Mean: %.4f", df["test_name_similarity"].mean())
        self.logger.info("  Median: %.4f", df["test_name_similarity"].median())

        self.logger.info("")
        self.logger.info("Distribuição das novas features por label:")
        
        triggers = df[df["label"] == 1]
        non_triggers = df[df["label"] == 0]

        self.logger.info("Triggers (label=1):")
        self.logger.info("  historical_failure_rate: mean=%.4f, median=%.4f",
                        triggers["historical_failure_rate"].mean(),
                        triggers["historical_failure_rate"].median())
        self.logger.info("  last_failure_distance: mean=%.2f, median=%.1f",
                        triggers["last_failure_distance"].mean(),
                        triggers["last_failure_distance"].median())
        self.logger.info("  test_name_similarity: mean=%.4f, median=%.4f",
                        triggers["test_name_similarity"].mean(),
                        triggers["test_name_similarity"].median())

        self.logger.info("Non-triggers (label=0):")
        self.logger.info("  historical_failure_rate: mean=%.4f, median=%.4f",
                        non_triggers["historical_failure_rate"].mean(),
                        non_triggers["historical_failure_rate"].median())
        self.logger.info("  last_failure_distance: mean=%.2f, median=%.1f",
                        non_triggers["last_failure_distance"].mean(),
                        non_triggers["last_failure_distance"].median())
        self.logger.info("  test_name_similarity: mean=%.4f, median=%.4f",
                        non_triggers["test_name_similarity"].mean(),
                        non_triggers["test_name_similarity"].median())
