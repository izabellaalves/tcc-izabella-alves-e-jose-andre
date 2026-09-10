"""Validação do arquivo features.csv."""

import pandas as pd
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger


class FeaturesValidator:
    """Valida o arquivo features.csv gerado pelo pipeline."""

    def __init__(self):
        self.logger = get_logger()
        self.validation_results: Dict[str, Any] = {}

    def validate_features_csv(self, csv_path: Path) -> bool:
        """Executa todas as validações no features.csv."""
        if not csv_path.exists():
            self.logger.error("Arquivo não encontrado: %s", csv_path)
            return False

        self.logger.info("Iniciando validação do features.csv")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.logger.error("Erro ao ler CSV: %s", e)
            return False

        all_passed = True
        all_passed &= self._validate_columns(df)
        all_passed &= self._validate_no_nulls(df)
        all_passed &= self._validate_bug_counts(df)
        all_passed &= self._validate_no_duplicates(df)
        all_passed &= self._validate_labels(df)
        all_passed &= self._validate_features(df)

        self._generate_report(df)

        if all_passed:
            self.logger.info("Todas as validações passaram")
        else:
            self.logger.warning("Algumas validações falharam")

        return all_passed

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        required_columns = [
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

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            self.logger.error("Colunas faltando: %s", missing)
            return False

        self.logger.info("Colunas obrigatórias presentes")
        return True

    def _validate_no_nulls(self, df: pd.DataFrame) -> bool:
        nulls = df.isnull().sum()
        has_nulls = nulls.sum() > 0

        if has_nulls:
            self.logger.error("Valores nulos encontrados:\n%s", nulls[nulls > 0])
            return False

        self.logger.info("Nenhum valor nulo encontrado")
        return True

    def _validate_bug_counts(self, df: pd.DataFrame) -> bool:
        expected = {"Lang": 61, "Chart": 26}
        actual = df.groupby("project")["bug"].nunique().to_dict()

        all_correct = True
        for project, expected_count in expected.items():
            actual_count = actual.get(project, 0)

            if actual_count != expected_count:
                self.logger.error(
                    "%s: esperado %d bugs, encontrado %d",
                    project,
                    expected_count,
                    actual_count,
                )
                all_correct = False
            else:
                self.logger.info(
                    "%s: %d/%d bugs processados",
                    project,
                    actual_count,
                    expected_count,
                )

        self.validation_results["bug_counts"] = actual
        return all_correct

    def _validate_no_duplicates(self, df: pd.DataFrame) -> bool:
        key_columns = ["project", "bug", "test_class", "test_method"]
        duplicates = df.duplicated(subset=key_columns, keep=False)
        num_duplicates = duplicates.sum()

        if num_duplicates > 0:
            self.logger.error("%d linhas duplicadas encontradas", num_duplicates)
            duplicate_rows = df[duplicates].sort_values(key_columns).head(10)
            self.logger.error("Exemplos:\n%s", duplicate_rows[key_columns].to_string())
            return False

        self.logger.info("Nenhuma duplicata encontrada")
        self.validation_results["duplicates"] = 0
        return True

    def _validate_labels(self, df: pd.DataFrame) -> bool:
        label_counts = df["label"].value_counts().to_dict()

        self.logger.info("Distribuição de labels:")
        for label, count in sorted(label_counts.items()):
            pct = count / len(df) * 100
            self.logger.info("  label=%s: %d (%.2f%%)", label, count, pct)

        self.validation_results["label_distribution"] = label_counts

        if label_counts.get(1, 0) == 0:
            self.logger.error("Nenhum caso positivo (label=1) encontrado")
            return False

        self.logger.info("Labels validados")
        return True

    def _validate_features(self, df: pd.DataFrame) -> bool:
        self.logger.info("Feature 'history':")
        history_stats = df["history"].describe()
        self.logger.info("  Min: %.0f", history_stats["min"])
        self.logger.info("  Max: %.0f", history_stats["max"])
        self.logger.info("  Mean: %.2f", history_stats["mean"])
        self.logger.info("  Median: %.2f", history_stats["50%"])
        self.validation_results["history_stats"] = history_stats.to_dict()

        self.logger.info("Feature 'same_package':")
        same_pkg_counts = df["same_package"].value_counts().to_dict()
        for value, count in sorted(same_pkg_counts.items()):
            pct = count / len(df) * 100
            self.logger.info("  %s: %d (%.2f%%)", value, count, pct)
        self.validation_results["same_package_distribution"] = same_pkg_counts

        self.logger.info("Feature 'modified_classes_count':")
        mod_classes_stats = df["modified_classes_count"].describe()
        self.logger.info("  Min: %.0f", mod_classes_stats["min"])
        self.logger.info("  Max: %.0f", mod_classes_stats["max"])
        self.logger.info("  Mean: %.2f", mod_classes_stats["mean"])
        self.logger.info("  Median: %.2f", mod_classes_stats["50%"])

        mod_classes_counts = df["modified_classes_count"].value_counts().head(10).to_dict()
        self.logger.info("  Distribuição (top 10):")
        for value, count in sorted(mod_classes_counts.items()):
            pct = count / len(df) * 100
            self.logger.info("    %s: %d (%.2f%%)", value, count, pct)

        self.validation_results["modified_classes_count_stats"] = mod_classes_stats.to_dict()
        self.validation_results["modified_classes_count_distribution"] = mod_classes_counts

        self.logger.info("Feature 'historical_failure_rate':")
        hfr_stats = df["historical_failure_rate"].describe()
        self.logger.info("  Min: %.4f", hfr_stats["min"])
        self.logger.info("  Max: %.4f", hfr_stats["max"])
        self.logger.info("  Mean: %.4f", hfr_stats["mean"])
        self.logger.info("  Median: %.4f", hfr_stats["50%"])
        self.validation_results["historical_failure_rate_stats"] = hfr_stats.to_dict()

        self.logger.info("Feature 'last_failure_distance':")
        lfd_stats = df["last_failure_distance"].describe()
        self.logger.info("  Min: %.0f", lfd_stats["min"])
        self.logger.info("  Max: %.0f", lfd_stats["max"])
        self.logger.info("  Mean: %.2f", lfd_stats["mean"])
        self.logger.info("  Median: %.0f", lfd_stats["50%"])
        self.validation_results["last_failure_distance_stats"] = lfd_stats.to_dict()

        self.logger.info("Feature 'test_name_similarity':")
        tns_stats = df["test_name_similarity"].describe()
        self.logger.info("  Min: %.4f", tns_stats["min"])
        self.logger.info("  Max: %.4f", tns_stats["max"])
        self.logger.info("  Mean: %.4f", tns_stats["mean"])
        self.logger.info("  Median: %.4f", tns_stats["50%"])
        self.validation_results["test_name_similarity_stats"] = tns_stats.to_dict()

        return True

    def _generate_report(self, df: pd.DataFrame):
        """Gera relatório final de validação."""
        total_rows = len(df)

        self.logger.info("=" * 80)
        self.logger.info("RELATÓRIO DE VALIDAÇÃO DO FEATURES.CSV")
        self.logger.info("=" * 80)
        self.logger.info("Total de linhas: %d", total_rows)

        self.logger.info("Bugs processados por projeto:")
        for project, count in df.groupby("project")["bug"].nunique().items():
            self.logger.info("  %s: %d bugs", project, count)

        self.logger.info("Distribuição de labels:")
        for label, count in sorted(df["label"].value_counts().items()):
            pct = count / total_rows * 100
            name = "positivos" if label == 1 else "negativos"
            self.logger.info("  %s (label=%s): %d (%.2f%%)", name, label, count, pct)

        duplicates = df.duplicated(
            subset=["project", "bug", "test_class", "test_method"]
        ).sum()
        self.logger.info("Duplicatas encontradas: %d", duplicates)

        self.logger.info("Feature 'history':")
        self.logger.info("  Min: %s", df["history"].min())
        self.logger.info("  Max: %s", df["history"].max())
        self.logger.info("  Mean: %.2f", df["history"].mean())
        self.logger.info("  Median: %.2f", df["history"].median())

        self.logger.info("Feature 'same_package':")
        for value, count in sorted(df["same_package"].value_counts().items()):
            pct = count / total_rows * 100
            self.logger.info("  %s: %d (%.2f%%)", value, count, pct)

        self.logger.info("Feature 'modified_classes_count':")
        self.logger.info("  Min: %s", df["modified_classes_count"].min())
        self.logger.info("  Max: %s", df["modified_classes_count"].max())
        self.logger.info("  Mean: %.2f", df["modified_classes_count"].mean())
        self.logger.info("  Median: %.2f", df["modified_classes_count"].median())

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

        if df["modified_classes_count"].nunique() == 1:
            self.logger.info(
                "Observação: todos os bugs possuem o mesmo modified_classes_count"
            )

        chart_bugs = df[df["project"] == "Chart"]["bug"].nunique()
        if chart_bugs < 26:
            self.logger.info(
                "Observação: apenas %d/26 bugs do Chart foram processados",
                chart_bugs,
            )

        self.logger.info("=" * 80)
