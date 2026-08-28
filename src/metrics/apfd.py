"""Métrica APFD (Average Percentage of Faults Detected).

Reutilizada pelas baselines (Random, History-Based) e pelo Random Forest.
"""

import warnings

import numpy as np


def calculate_apfd(ordered_labels) -> float:
    """Calcula o APFD de uma ordenação de testes para um bug.

    ordered_labels: sequência de labels (0/1) na ordem de execução proposta.
    Cada bug do Defects4J representa uma única falha (m=1), então:

        APFD = 1 - (TF_1 / n) + 1 / (2n)

    onde n é o total de testes e TF_1 a posição (1-indexada) do primeiro
    teste com label=1. Retorna NaN (com aviso) se não houver trigger test.
    """
    labels = np.asarray(ordered_labels)
    n = len(labels)
    trigger_positions = np.flatnonzero(labels == 1)
    if n == 0 or len(trigger_positions) == 0:
        warnings.warn("Nenhum trigger test (label=1) na ordenação; APFD indefinido.")
        return float("nan")
    tf1 = trigger_positions[0] + 1  # posição 1-indexada
    return 1 - tf1 / n + 1 / (2 * n)


if __name__ == "__main__":
    # Self-check
    assert calculate_apfd([1, 0, 0, 0]) == 1 - 1 / 4 + 1 / 8      # melhor caso
    assert calculate_apfd([0, 0, 0, 1]) == 1 - 4 / 4 + 1 / 8      # pior caso
    assert calculate_apfd([0, 1, 1, 0]) == 1 - 2 / 4 + 1 / 8      # usa o primeiro trigger
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(calculate_apfd([0, 0, 0]))                 # sem trigger
    print("apfd.py: todos os self-checks passaram.")
