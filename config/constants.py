"""Constantes do projeto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERMEDIATE_DIR = DATA_DIR / "intermediate"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_RESULTS_DIR = DATA_DIR / "results"

DEFECTS4J_DIR = PROJECT_ROOT / "defects4j"

PROJECTS = ["Lang", "Chart", "Math", "Time", "Mockito", "Compress"]
EXPECTED_BUGS = {
    "Lang": 58,
    "Chart": 26,
    "Math": 102,
    "Time": 25,
    "Mockito": 31,
    "Compress": 46,
}

DEFECTS4J_PROPERTIES = [
    "tests.trigger",
    "tests.relevant",
    "classes.modified",
    "dir.bin.tests",
]

FEATURE_COLUMNS = [
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

CHECKOUT_TIMEOUT = 600
COMPILE_TIMEOUT = 600
EXPORT_TIMEOUT = 60
JAVAP_TIMEOUT = 30

REQUIRED_JAVA_VERSION = 11
DEFECTS4J_VERSION = "3.0.1"
