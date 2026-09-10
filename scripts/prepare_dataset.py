#!/usr/bin/env python3
"""Pipeline de Feature Engineering para priorização de testes."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import DATA_INTERMEDIATE_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.defects4j.checkout import CheckoutManager
from src.defects4j.metadata_exporter import MetadataExporter
from src.defects4j.wrapper import Defects4JWrapper
from src.feature_engineering.engineer import FeatureEngineer
from src.feature_engineering.test_enumerator import TestEnumerator
from src.utils.environment import validate_environment
from src.utils.helpers import Timer, format_duration
from src.utils.logger import setup_logger
from src.utils.validation import FeaturesValidator


def parse_arguments():
    """Parseia argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Feature Engineering para TCC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python prepare_dataset.py
  python prepare_dataset.py --projects Lang
  python prepare_dataset.py --skip-checkout --debug
        """,
    )

    parser.add_argument(
        "--projects",
        type=str,
        default="Lang,Chart,Math,Time,Mockito,Compress",
        help="Projetos a processar, separados por vírgula (padrão: Lang,Chart,Math,Time,Mockito,Compress)",
    )
    parser.add_argument(
        "--skip-checkout",
        action="store_true",
        help="Pular checkout se bugs já existirem em data/",
    )
    parser.add_argument(
        "--skip-to-features",
        action="store_true",
        help="Pular direto para cálculo de features usando intermediate.csv existente",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativar logging em nível DEBUG",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Diretório para checkouts (padrão: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Arquivo de saída (padrão: data/processed/features.csv)",
    )

    return parser.parse_args()


def main():
    """Executa o pipeline completo de feature engineering."""
    args = parse_arguments()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_RAW_DIR
    output_path = Path(args.output) if args.output else DATA_PROCESSED_DIR / "features.csv"

    log_level = 10 if args.debug else 20
    logger = setup_logger(level=log_level)

    logger.info("=" * 80)
    logger.info("TCC - Pipeline de Feature Engineering")
    logger.info("=" * 80)
    logger.info("Início: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Projetos: %s", ", ".join(p.strip() for p in args.projects.split(",")))
    logger.info("Skip checkout: %s", args.skip_checkout)
    logger.info("Skip to features: %s", args.skip_to_features)
    logger.info("Diretório de dados: %s", data_dir)
    logger.info("Arquivo de saída: %s", output_path)
    logger.info("")

    projects = [p.strip() for p in args.projects.split(",")]
    total_timer = Timer()
    total_timer.__enter__()

    try:
        if args.skip_to_features:
            logger.info("=" * 80)
            logger.info("MODO RÁPIDO: Pulando para Cálculo de Features")
            logger.info("=" * 80)
            
            intermediate_path = DATA_INTERMEDIATE_DIR / "intermediate.csv"
            
            if not intermediate_path.exists():
                logger.error("Arquivo intermediate.csv não encontrado: %s", intermediate_path)
                logger.error("Execute primeiro sem --skip-to-features para gerar o arquivo")
                return 1
            
            logger.info("Carregando %s", intermediate_path)
            import pandas as pd
            df_intermediate = pd.read_csv(intermediate_path)
            logger.info("Carregado: %d linhas", len(df_intermediate))
            logger.info("")
            
            try:
                config = validate_environment()
            except RuntimeError as e:
                logger.error("Erro de ambiente: %s", e)
                return 1
            
            feature_engineer = FeatureEngineer(config)
            
            logger.info("=" * 80)
            logger.info("ETAPA 6: Cálculo de Features")
            logger.info("=" * 80)

            with Timer() as t:
                df_features = feature_engineer.calculate_features(
                    df_intermediate,
                    [],
                    {},
                )

            logger.info("Features calculadas em %s", format_duration(t.elapsed))
            logger.info("")
            
            logger.info("=" * 80)
            logger.info("ETAPA 7: Validação de Features")
            logger.info("=" * 80)

            if not feature_engineer.validate_features(df_features):
                logger.error("Validação de features falhou")
                return 1

            logger.info("")

            logger.info("=" * 80)
            logger.info("ETAPA 8: Estatísticas Finais")
            logger.info("=" * 80)

            feature_engineer.print_statistics(df_features)

            DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            df_features.to_csv(output_path, index=False)

            logger.info("")
            logger.info("=" * 80)
            logger.info("Pipeline Finalizado (MODO RÁPIDO)")
            logger.info("=" * 80)
            logger.info("Arquivo salvo: %s", output_path)
            logger.info("Linhas: %d", len(df_features))
            logger.info("Tempo total: %s", format_duration(total_timer.elapsed))
            
            return 0

        logger.info("=" * 80)
        logger.info("ETAPA 1: Validação do Ambiente")
        logger.info("=" * 80)

        try:
            config = validate_environment()
        except RuntimeError as e:
            logger.error("Erro de ambiente: %s", e)
            return 1

        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 2: Checkout e Compilação de Bugs")
        logger.info("=" * 80)

        checkout_manager = CheckoutManager(config, data_dir)

        with Timer() as t:
            bug_infos = checkout_manager.process_multiple_projects(
                projects,
                skip_if_exists=args.skip_checkout,
            )

        if not bug_infos:
            logger.error("Nenhum bug foi processado com sucesso")
            return 1

        logger.info("Checkout concluído em %s", format_duration(t.elapsed))
        logger.info("Bugs prontos: %d", len(bug_infos))
        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 3: Exportação de Metadados")
        logger.info("=" * 80)

        wrapper = Defects4JWrapper(config)
        metadata_exporter = MetadataExporter(wrapper)

        with Timer() as t:
            metadatas = metadata_exporter.export_multiple_bugs(bug_infos)

        if not metadatas:
            logger.error("Nenhum metadado foi exportado com sucesso")
            return 1

        logger.info("Metadados exportados em %s", format_duration(t.elapsed))
        logger.info("Bugs com metadados: %d", len(metadatas))
        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 4: Enumeração de Métodos de Teste")
        logger.info("=" * 80)

        test_enumerator = TestEnumerator(config)

        with Timer() as t:
            test_methods_dict = test_enumerator.enumerate_multiple_bugs(metadatas)

        total_methods = sum(len(methods) for methods in test_methods_dict.values())

        if total_methods == 0:
            logger.error("Nenhum método de teste foi enumerado")
            return 1

        logger.info("Métodos enumerados em %s", format_duration(t.elapsed))
        logger.info("Total de métodos: %d", total_methods)
        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 5: Construção da Tabela Intermediária")
        logger.info("=" * 80)

        feature_engineer = FeatureEngineer(config)

        with Timer() as t:
            df_intermediate = feature_engineer.build_intermediate_table(
                metadatas,
                test_methods_dict,
            )

        logger.info("Tabela intermediária construída em %s", format_duration(t.elapsed))

        DATA_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        intermediate_path = DATA_INTERMEDIATE_DIR / "intermediate.csv"
        df_intermediate.to_csv(intermediate_path, index=False)
        logger.info("Salvo: %s", intermediate_path)
        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 6: Cálculo de Features")
        logger.info("=" * 80)

        with Timer() as t:
            df_features = feature_engineer.calculate_features(
                df_intermediate,
                metadatas,
                test_methods_dict,
            )

        logger.info("Features calculadas em %s", format_duration(t.elapsed))
        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 7: Validação de Features")
        logger.info("=" * 80)

        if not feature_engineer.validate_features(df_features):
            logger.error("Validação falhou")
            return 1

        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 8: Salvando features.csv")
        logger.info("=" * 80)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False)

        logger.info("Arquivo salvo: %s", output_path)
        logger.info("Tamanho: %.2f KB", output_path.stat().st_size / 1024)
        logger.info("Linhas: %d", len(df_features))
        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 9: Estatísticas Finais")
        logger.info("=" * 80)

        feature_engineer.print_statistics(df_features)
        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 10: Validação Completa do features.csv")
        logger.info("=" * 80)

        validator = FeaturesValidator()
        validation_passed = validator.validate_features_csv(output_path)

        if not validation_passed:
            logger.warning("Algumas validações falharam, mas o arquivo foi gerado")

        logger.info("")

        logger.info("=" * 80)
        logger.info("ETAPA 11: Validação da Feature history_detection")
        logger.info("=" * 80)

        from validate_history_feature import HistoryValidator

        history_validator = HistoryValidator(output_path)
        history_validation_passed = history_validator.run_all_validations()

        if not history_validation_passed:
            logger.error("VALIDAÇÃO DE HISTORY FALHOU!")
            logger.error("O dataset NÃO está pronto para treinamento.")
            return 1

        logger.info("")

        total_timer.__exit__(None, None, None)

        logger.info("=" * 80)
        logger.info("Pipeline concluído com sucesso")
        logger.info("=" * 80)
        logger.info("Tempo total: %s", format_duration(total_timer.elapsed))
        logger.info("Arquivo final: %s", output_path.absolute())
        logger.info("Fim: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("Pipeline interrompido pelo usuário")
        return 130

    except Exception as e:
        logger.exception("Erro fatal durante execução: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
