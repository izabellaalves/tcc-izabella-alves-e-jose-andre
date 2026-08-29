#!/usr/bin/env python3
"""
Validação da feature history_detection no CSV final.

Este script DEVE ser executado APÓS prepare_dataset.py e ANTES de treinar o modelo.
Se qualquer validação falhar, o treinamento NÃO deve prosseguir.

Uso:
    python scripts/validate_history_feature.py
    python scripts/validate_history_feature.py --csv data/processed/features.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger


class HistoryValidator:
    """Valida a feature history_detection no CSV de features."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.logger = setup_logger()
        self.df = None
        self.errors = []
        self.warnings = []

    def load_csv(self) -> bool:
        """Carrega o CSV de features."""
        if not self.csv_path.exists():
            self.logger.error("CSV não encontrado: %s", self.csv_path)
            return False

        try:
            self.df = pd.read_csv(self.csv_path)
            self.logger.info("CSV carregado: %d linhas", len(self.df))
            return True
        except Exception as e:
            self.logger.error("Erro ao carregar CSV: %s", e)
            return False

    def validate_basic_properties(self) -> bool:
        """Valida propriedades básicas do history."""
        self.logger.info("Validando propriedades básicas...")

        valid = True

        # 1. history deve ser >= 0
        negative_count = (self.df["history"] < 0).sum()
        if negative_count > 0:
            self.errors.append(
                f"Encontrados {negative_count} valores negativos em history"
            )
            valid = False

        # 2. history deve ser inteiro
        if self.df["history"].dtype not in ["int64", "int32"]:
            self.errors.append(
                f"history deveria ser int, encontrado: {self.df['history'].dtype}"
            )
            valid = False

        # 3. Verificar distribuição
        self.logger.info("Distribuição de history:")
        for val, count in self.df["history"].value_counts().sort_index().head(10).items():
            pct = count / len(self.df) * 100
            self.logger.info("  history=%d: %d (%.2f%%)", val, count, pct)

        return valid

    def validate_known_test_cases(self) -> bool:
        """Valida casos conhecidos de testes que detectam múltiplos bugs."""
        self.logger.info("Validando casos conhecidos...")

        valid = True

        # Caso 1: testCreateNumber detecta 4 bugs
        test_name = "testCreateNumber"
        test_class = "org.apache.commons.lang3.math.NumberUtilsTest"
        expected_bugs = [7, 16, 27, 36]
        expected_histories = [0, 1, 2, 3]

        test_df = self.df[
            (self.df["test_class"] == test_class) & (self.df["test_method"] == test_name)
        ].sort_values("bug")

        positives = test_df[test_df["label"] == 1]

        if len(positives) == 0:
            self.warnings.append(
                f"Teste {test_name} não encontrado ou não detecta nenhum bug"
            )
        else:
            actual_bugs = positives["bug"].tolist()
            actual_histories = positives["history"].tolist()

            self.logger.info(
                "Teste: %s::%s",
                test_class.split(".")[-1],
                test_name,
            )
            self.logger.info("  Bugs detectados: %s", actual_bugs)
            self.logger.info("  Histories: %s", actual_histories)

            if actual_bugs != expected_bugs:
                self.warnings.append(
                    f"{test_name}: bugs esperados {expected_bugs}, encontrados {actual_bugs}"
                )

            if actual_histories != expected_histories:
                self.errors.append(
                    f"{test_name}: histories esperados {expected_histories}, "
                    f"encontrados {actual_histories} "
                )
                valid = False
            else:
                self.logger.info("  [OK] Histories corretos!")

        # Caso 2: testIsNumber detecta 2 bugs
        test_name = "testIsNumber"
        test_class = "org.apache.commons.lang3.math.NumberUtilsTest"

        test_df = self.df[
            (self.df["test_class"] == test_class) & (self.df["test_method"] == test_name)
        ].sort_values("bug")

        positives = test_df[test_df["label"] == 1]

        if len(positives) >= 2:
            actual_histories = positives["history"].tolist()
            expected_histories = list(range(len(positives)))  # [0, 1, ...] baseado em quantos bugs detecta

            self.logger.info(
                "Teste: %s::%s",
                test_class.split(".")[-1],
                test_name,
            )
            self.logger.info("  Bugs detectados: %s", positives["bug"].tolist())
            self.logger.info("  Histories: %s", actual_histories)

            if actual_histories != expected_histories:
                self.errors.append(
                    f"{test_name}: histories esperados {expected_histories}, "
                    f"encontrados {actual_histories} "
                )
                valid = False
            else:
                self.logger.info("  [OK] Histories corretos!")

        return valid

    def validate_history_consistency(self) -> bool:
        """
        Para cada teste, valida que o history é consistente:
        history no bug atual = número de bugs anteriores detectados pelo mesmo teste.
        """
        self.logger.info("Validando consistência de history para todos os testes...")

        valid = True
        inconsistencies = []

        # Agrupar por projeto e teste
        for (project, test_class, test_method), group in self.df.groupby(
            ["project", "test_class", "test_method"]
        ):
            # Ordenar por bug
            group = group.sort_values("bug")

            # Calcular o history esperado manualmente
            bugs_detected_before = []

            for idx, row in group.iterrows():
                expected_history = len(bugs_detected_before)
                actual_history = row["history"]

                if expected_history != actual_history:
                    inconsistencies.append(
                        {
                            "project": project,
                            "test": f"{test_class.split('.')[-1]}::{test_method}",
                            "bug": row["bug"],
                            "expected": expected_history,
                            "actual": actual_history,
                            "label": row["label"],
                        }
                    )

                # Se detectou o bug, adicionar à lista
                if row["label"] == 1:
                    bugs_detected_before.append(row["bug"])

        if inconsistencies:
            self.logger.error("Encontradas %d inconsistências em history!", len(inconsistencies))

            # Mostrar alguns exemplos
            for inc in inconsistencies[:10]:
                self.logger.error(
                    "  %s bug %d: esperado history=%d, encontrado history=%d (label=%d)",
                    inc["test"],
                    inc["bug"],
                    inc["expected"],
                    inc["actual"],
                    inc["label"],
                )

            if len(inconsistencies) > 10:
                self.logger.error("  ... e mais %d inconsistências", len(inconsistencies) - 10)

            self.errors.append(f"{len(inconsistencies)} inconsistências em history")
            valid = False
        else:
            self.logger.info("[OK] Todos os valores de history sao consistentes!")

        return valid

    def validate_positive_distribution(self) -> bool:
        """Valida que positivos têm distribuição razoável de history."""
        self.logger.info("Validando distribuição de history em positivos...")

        positives = self.df[self.df["label"] == 1]
        total_positives = len(positives)

        history_counts = positives["history"].value_counts().sort_index()

        self.logger.info("Distribuição de history em positivos:")
        for val, count in history_counts.items():
            pct = count / total_positives * 100
            self.logger.info("  history=%d: %d (%.2f%%)", val, count, pct)

        # Verificar se 100% tem history=0 (sinal de bug)
        history_0_pct = (history_counts.get(0, 0) / total_positives) * 100

        if history_0_pct == 100.0:
            self.errors.append(
                "100% dos positivos têm history=0 - provável bug de implementação!"
            )
            return False
        elif history_0_pct > 99.0:
            self.warnings.append(
                f"{history_0_pct:.1f}% dos positivos têm history=0 - verificar se está correto"
            )

        return True

    def validate_negative_distribution(self) -> bool:
        """Valida distribuição de history em negativos."""
        self.logger.info("Validando distribuição de history em negativos...")

        negatives = self.df[self.df["label"] == 0]
        total_negatives = len(negatives)

        history_gt_0 = (negatives["history"] > 0).sum()
        pct_gt_0 = (history_gt_0 / total_negatives) * 100

        self.logger.info(
            "Negativos com history > 0: %d (%.2f%%)",
            history_gt_0,
            pct_gt_0,
        )

        # Deve haver alguns negativos com history > 0
        # (testes que detectaram bugs antes, mas não detectam o bug atual)
        if history_gt_0 == 0:
            self.warnings.append("Nenhum negativo tem history > 0 - pode indicar problema")

        return True

    def validate_history_per_project(self) -> bool:
        """Valida que history é separado por projeto."""
        self.logger.info("Validando separação de history por projeto...")

        valid = True

        for project in self.df["project"].unique():
            project_df = self.df[self.df["project"] == project]

            max_history = project_df["history"].max()
            positives_count = project_df["label"].sum()

            self.logger.info(
                "%s: max_history=%d, positives=%d",
                project,
                max_history,
                positives_count,
            )

            # max_history não pode ser maior que positives_count - 1
            # (se um teste detecta N bugs, max history é N-1)
            if max_history >= positives_count:
                self.warnings.append(
                    f"{project}: max_history={max_history} >= positives={positives_count}"
                )

        return valid

    def run_all_validations(self) -> bool:
        """Executa todas as validações."""
        self.logger.info("=" * 80)
        self.logger.info("VALIDAÇÃO DA FEATURE history_detection")
        self.logger.info("=" * 80)

        if not self.load_csv():
            return False

        validations = [
            ("Propriedades básicas", self.validate_basic_properties),
            ("Casos conhecidos", self.validate_known_test_cases),
            ("Consistência de history", self.validate_history_consistency),
            ("Distribuição em positivos", self.validate_positive_distribution),
            ("Distribuição em negativos", self.validate_negative_distribution),
            ("Separação por projeto", self.validate_history_per_project),
        ]

        all_valid = True

        for name, validation_func in validations:
            self.logger.info("")
            self.logger.info("-" * 80)
            try:
                result = validation_func()
                if not result:
                    all_valid = False
            except Exception as e:
                self.logger.exception("Erro durante validação '%s': %s", name, e)
                self.errors.append(f"Erro em '{name}': {e}")
                all_valid = False

        # Relatório final
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("RESULTADO DA VALIDAÇÃO")
        self.logger.info("=" * 80)

        if self.warnings:
            self.logger.warning("Avisos (%d):", len(self.warnings))
            for warning in self.warnings:
                self.logger.warning("  [!] %s", warning)

        if self.errors:
            self.logger.error("Erros (%d):", len(self.errors))
            for error in self.errors:
                self.logger.error("  [ERRO] %s", error)

        if all_valid and not self.errors:
            self.logger.info("[OK] TODAS AS VALIDACOES PASSARAM!")
            self.logger.info("")
            self.logger.info("O dataset esta correto e pode ser usado para treinamento.")
            return True
        else:
            self.logger.error("[FALHOU] VALIDACAO FALHOU!")
            self.logger.error("")
            self.logger.error("NAO prossiga com o treinamento ate corrigir os erros.")
            return False


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Valida a feature history_detection no CSV de features"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/features.csv",
        help="Caminho para o CSV de features (padrão: data/processed/features.csv)",
    )

    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_arguments()
    csv_path = Path(args.csv)

    validator = HistoryValidator(csv_path)
    success = validator.run_all_validations()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
