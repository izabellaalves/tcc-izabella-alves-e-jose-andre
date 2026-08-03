"""Integração com Defects4J."""

from .checkout import BugInfo, CheckoutManager
from .metadata_exporter import BugMetadata, MetadataExporter
from .wrapper import CommandResult, Defects4JWrapper

__all__ = [
    "Defects4JWrapper",
    "CommandResult",
    "CheckoutManager",
    "BugInfo",
    "MetadataExporter",
    "BugMetadata",
]
