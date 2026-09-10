#!/usr/bin/env python3
"""Recalcula apenas as features a partir do intermediate.csv existente."""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.engineer import FeatureEngineer
from src.utils.environment import EnvironmentConfig

# Config mínima sem validação
config = EnvironmentConfig(
    env_type="local",
    java_version="11",
    java_home=Path("/usr/lib/jvm/java-11"),
    python_version="3.12",
    defects4j_path=Path("defects4j"),
    defects4j_cmd="defects4j",
    perl_available=True,
    git_available=True
)

print("Carregando intermediate.csv...")
df_intermediate = pd.read_csv("data/intermediate/intermediate.csv")
print(f"Carregado: {len(df_intermediate)} linhas")

print("\nRecalculando features...")
engineer = FeatureEngineer(config)
df_features = engineer.calculate_features(df_intermediate, [], {})

print(f"\nValidando features...")
if engineer.validate_features(df_features):
    print("Validação OK!")
else:
    print("ERRO na validação!")
    sys.exit(1)

print("\nSalvando features.csv...")
df_features.to_csv("data/processed/features.csv", index=False)

print("\nEstatísticas:")
engineer.print_statistics(df_features)

print(f"\nSucesso! {len(df_features)} linhas salvas em data/processed/features.csv")
