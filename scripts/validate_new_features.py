#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger


class NewFeaturesValidator:

    def __init__(self, features_csv: Path):
        self.features_csv = features_csv
        self.logger = get_logger()
        self.df = None

    def load_data(self) -> bool:
        if not self.features_csv.exists():
            self.logger.error("Arquivo não encontrado: %s", self.features_csv)
            return False

        try:
            self.df = pd.read_csv(self.features_csv)
            self.logger.info("Dataset carregado: %d linhas", len(self.df))
            return True
        except Exception as e:
            self.logger.error("Erro ao carregar CSV: %s", e)
            return False

    def validate_historical_failure_rate(self) -> bool:
        self.logger.info("=" * 80)
        self.logger.info("Validando historical_failure_rate")
        self.logger.info("=" * 80)
        
        errors = []

        if (self.df["historical_failure_rate"] < 0).any():
            errors.append("historical_failure_rate tem valores < 0")
        
        if (self.df["historical_failure_rate"] > 1).any():
            errors.append("historical_failure_rate tem valores > 1")

        null_count = self.df["historical_failure_rate"].isnull().sum()
        if null_count > 0:
            errors.append(f"historical_failure_rate tem {null_count} valores nulos")

        first_bugs = self.df.groupby("project")["bug"].min()
        for project, first_bug in first_bugs.items():
            first_bug_df = self.df[(self.df["project"] == project) & (self.df["bug"] == first_bug)]
            if (first_bug_df["historical_failure_rate"] != 0.0).any():
                errors.append(f"Primeiro bug {project}-{first_bug} tem historical_failure_rate != 0")

        if errors:
            for error in errors:
                self.logger.error(error)
            return False

        self.logger.info("Min: %.4f", self.df["historical_failure_rate"].min())
        self.logger.info("Max: %.4f", self.df["historical_failure_rate"].max())
        self.logger.info("Mean: %.4f", self.df["historical_failure_rate"].mean())
        self.logger.info("Median: %.4f", self.df["historical_failure_rate"].median())

        comparison_df = self.df[["project", "bug", "test_class", "test_method", "history", "historical_failure_rate"]].groupby(
            ["project", "bug", "test_class", "test_method"]
        ).first().reset_index()
        
        self.logger.info("\nComparação history vs historical_failure_rate (primeiras 5 linhas com history > 0):")
        sample = comparison_df[comparison_df["history"] > 0].head(5)
        for _, row in sample.iterrows():
            self.logger.info(
                "  %s-%s %s::%s - history=%d, hfr=%.4f",
                row["project"],
                row["bug"],
                row["test_class"],
                row["test_method"],
                row["history"],
                row["historical_failure_rate"],
            )

        self.logger.info("historical_failure_rate PASSOU")
        return True

    def validate_last_failure_distance(self) -> bool:
        self.logger.info("=" * 80)
        self.logger.info("Validando last_failure_distance")
        self.logger.info("=" * 80)
        
        errors = []

        if (self.df["last_failure_distance"] < 0).any():
            errors.append("last_failure_distance tem valores negativos")

        null_count = self.df["last_failure_distance"].isnull().sum()
        if null_count > 0:
            errors.append(f"last_failure_distance tem {null_count} valores nulos")

        if errors:
            for error in errors:
                self.logger.error(error)
            return False

        self.logger.info("Min: %d", self.df["last_failure_distance"].min())
        self.logger.info("Max: %d", self.df["last_failure_distance"].max())
        self.logger.info("Mean: %.2f", self.df["last_failure_distance"].mean())
        self.logger.info("Median: %.1f", self.df["last_failure_distance"].median())

        self.logger.info("\nExemplos de last_failure_distance por teste (primeiros 5 com distância > 0):")
        sample = self.df[self.df["last_failure_distance"] > 0].head(5)
        for _, row in sample.iterrows():
            self.logger.info(
                "  %s-%s %s::%s - distance=%d",
                row["project"],
                row["bug"],
                row["test_class"],
                row["test_method"],
                row["last_failure_distance"],
            )

        self.logger.info("last_failure_distance PASSOU")
        return True

    def validate_test_name_similarity(self) -> bool:
        self.logger.info("=" * 80)
        self.logger.info("Validando test_name_similarity")
        self.logger.info("=" * 80)
        
        errors = []

        if (self.df["test_name_similarity"] < 0).any():
            errors.append("test_name_similarity tem valores < 0")

        if (self.df["test_name_similarity"] > 1).any():
            errors.append("test_name_similarity tem valores > 1")

        null_count = self.df["test_name_similarity"].isnull().sum()
        if null_count > 0:
            errors.append(f"test_name_similarity tem {null_count} valores nulos")

        if errors:
            for error in errors:
                self.logger.error(error)
            return False

        self.logger.info("Min: %.4f", self.df["test_name_similarity"].min())
        self.logger.info("Max: %.4f", self.df["test_name_similarity"].max())
        self.logger.info("Mean: %.4f", self.df["test_name_similarity"].mean())
        self.logger.info("Median: %.4f", self.df["test_name_similarity"].median())

        high_sim = self.df[self.df["test_name_similarity"] > 0.5]
        self.logger.info("\nTestes com alta similaridade (>0.5): %d (%.2f%%)", 
                        len(high_sim), len(high_sim) / len(self.df) * 100)

        self.logger.info("\nExemplos de alta similaridade (primeiros 5):")
        sample = self.df[self.df["test_name_similarity"] > 0.5].head(5)
        for _, row in sample.iterrows():
            self.logger.info(
                "  %s-%s %s - similarity=%.4f",
                row["project"],
                row["bug"],
                row["test_class"],
                row["test_name_similarity"],
            )

        self.logger.info("test_name_similarity PASSOU")
        return True

    def run_all_validations(self) -> bool:
        self.logger.info("=" * 80)
        self.logger.info("Validação das Novas Features")
        self.logger.info("=" * 80)
        self.logger.info("")

        if not self.load_data():
            return False

        validations = [
            ("historical_failure_rate", self.validate_historical_failure_rate),
            ("last_failure_distance", self.validate_last_failure_distance),
            ("test_name_similarity", self.validate_test_name_similarity),
        ]

        all_passed = True

        for name, validation_func in validations:
            self.logger.info("")
            try:
                if not validation_func():
                    self.logger.error("Validação '%s' FALHOU", name)
                    all_passed = False
            except Exception as e:
                self.logger.exception("Erro na validação '%s': %s", name, e)
                all_passed = False

        self.logger.info("")
        self.logger.info("=" * 80)

        if all_passed:
            self.logger.info("TODAS AS VALIDAÇÕES PASSARAM")
        else:
            self.logger.error("ALGUMAS VALIDAÇÕES FALHARAM")

        return all_passed


def main():
    features_csv = Path("data/processed/features.csv")
    
    validator = NewFeaturesValidator(features_csv)
    success = validator.run_all_validations()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
