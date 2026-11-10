"""Gemeinsame Fixtures: das Handbeispiel (sechs Container, zwei vorzeitig ausgeschieden), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAND_T = [2, 3, 4, 5, 5, 8]
HAND_E = [1, 0, 1, 1, 0, 1]      # 3+ und 5+ sind vorzeitig ausgeschieden


@pytest.fixture
def hand():
    """Zeiten 2, 3+, 4, 5, 5+, 8 (ein + heißt ausgeschieden). Von Hand: Kaplan–Meier 5/6, 5/8, 5/12, 0 (Median 5); untere Schranke (3 und 5+ gehen sofort ab) Median 4;
    obere Schranke (beide bleiben für immer) 5/6, 2/3, 1/2, 1/3 bei t = 2, 4, 5, 8 (Median 5)."""
    return list(HAND_T), list(HAND_E)
