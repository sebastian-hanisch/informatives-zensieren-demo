"""Tests des kopierten Kerns inz_core (aus ueberlebenszeit-planung/kern/test_uz_core.py, auf die hier genutzten Funktionen gekürzt und angepasst)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import inz_core as K

SIX_T = [2, 3, 4, 5, 5, 8]
SIX_E = [1, 0, 1, 1, 0, 1]


def test_km_handbeispiel_sechs_container():
    km = K.kaplan_meier(SIX_T, SIX_E)
    assert km["t"].tolist() == [2, 4, 5, 8]
    assert km["n"].tolist() == [6, 4, 3, 1]
    for got, want in zip(km["s"], [F(5, 6), F(5, 8), F(5, 12), F(0)]):
        assert got == pytest.approx(float(want), abs=1e-15)
    assert K.quantile(km, 0.5) == 5.0


def test_km_ohne_zensierung_ist_eins_minus_verteilungsfunktion():
    rng = K.SplitMix64(1)
    t = np.round(rng.lognormal(50, 4.0, 0.9), 0) + 1.0   # ganzzahlig, mit Gleichständen
    km = K.kaplan_meier(t, np.ones(50, bool))
    for x in (1.0, 2.0, 4.0, 8.0):
        assert K.survival_at(km, x) == pytest.approx(float(np.mean(t > x)), abs=1e-12)


def test_km_zensierung_hinter_letztem_ereignis():
    # Die ZEIT einer Zensierung hinter dem letzten Ereignis ändert die Kurve nicht ...
    a = K.kaplan_meier([2, 4, 6, 9], [1, 1, 1, 0])
    b = K.kaplan_meier([2, 4, 6, 20], [1, 1, 1, 0])
    assert np.array_equal(a["s"], b["s"])
    # ... eine ZUSÄTZLICHE Zensierung dort zählt aber in den früheren Risikomengen mit und hebt die Kurve.
    c = K.kaplan_meier([2, 4, 6], [1, 1, 1])
    d = K.kaplan_meier([2, 4, 6, 9, 11], [1, 1, 1, 0, 0])
    assert np.allclose(d["s"], [4 / 5, 4 / 5 * 3 / 4, 4 / 5 * 3 / 4 * 2 / 3])
    assert np.all(d["s"] > c["s"] - 1e-15) and d["s"][0] > c["s"][0]


def test_km_alle_zensiert():
    km = K.kaplan_meier([1, 2, 3], [0, 0, 0])
    assert km["t"].size == 0 and math.isnan(K.quantile(km, 0.5)) and K.survival_at(km, 5.0) == 1.0


def test_quantil_nicht_schaetzbar_wenn_kurve_nicht_faellt():
    km = K.kaplan_meier([1, 2, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0])
    assert math.isnan(K.quantile(km, 0.5))


def test_eingabepruefung():
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([3, 3, 5], [1, 1, 1], [3, 3, 0])  # Eintritt = Zeit
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([0, 1], [1, 1])  # Zeit 0
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1, 2])  # Ereignis 2
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1])  # Länge
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([], [])


def test_splitmix_referenzwert_und_determinismus():
    # erster Ausgabewert von SplitMix64 bei Seed 0 (Referenz: Vigna, splitmix64.c)
    assert int(K.SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF
    a, b = K.SplitMix64(5), K.SplitMix64(5)
    assert np.array_equal(a.uniform(10), b.uniform(10))
    u = K.SplitMix64(9).uniform(50000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01
    z = K.SplitMix64(9).normal(50000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02
    x = K.SplitMix64(9).lognormal(50000, 4.0, 0.9)
    assert abs(float(np.median(x)) - 4.0) < 0.1
