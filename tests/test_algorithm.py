"""Einheiten des Verfahrens an der Mini-Instanz von Hand (DEMO-PLAYBOOK §1b): untere und obere Schranke, gewichtete Kurve, Kipppunkt.

Handbeispiel (Zeiten 2, 3+, 4, 5, 5+, 8; + = vorzeitig ausgeschieden), Brüche von Hand gerechnet:
  Kaplan–Meier:   n = 6, 4, 3, 1  ->  S = 5/6, 5/8, 5/12, 0                           Median 5
  untere Schranke (3+ und 5+ gehen sofort ab, alle sechs sind Ereignisse):
                  t = 2: 5/6; t = 3: 5/6 · 4/5 = 2/3; t = 4: 2/3 · 3/4 = 1/2; t = 5 (zwei Ereignisse, n = 3): 1/2 · 1/3 = 1/6; t = 8: 0       Median 4
  obere Schranke (3+ und 5+ bleiben für immer):  t = 2: 5/6; t = 4: n = 5, 5/6 · 4/5 = 2/3; t = 5: n = 4, 2/3 · 3/4 = 1/2; t = 8: n = 3, 1/2 · 2/3 = 1/3      Median 5
  φ = 1/2 (jeder Ausgeschiedene zur Hälfte als Ereignis): t = 2: 5/6; t = 3: n = 5, d = 1/2, Faktor 9/10 -> 3/4; t = 4: n = 4, Faktor 3/4 -> 9/16;
                  t = 5: n = 3, d = 3/2, Faktor 1/2 -> 9/32; t = 8: 0                    Median 5"""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import inz_algorithm as A
import inz_core as K


def test_untere_schranke_handbeispiel(hand):
    t, e = hand
    lo = A.lower_bound_km(t, e)
    assert lo["t"].tolist() == [2, 3, 4, 5, 8]
    assert lo["n"].tolist() == [6, 5, 4, 3, 1] and lo["d"].tolist() == [1, 1, 1, 2, 1]
    for got, want in zip(lo["s"], [F(5, 6), F(2, 3), F(1, 2), F(1, 6), F(0)]):
        assert got == pytest.approx(float(want), abs=1e-15)
    assert K.quantile(lo, 0.5) == 4.0


def test_obere_schranke_handbeispiel(hand):
    t, e = hand
    up = A.upper_bound_km(t, e)
    assert up["t"].tolist() == [2, 4, 5, 8]
    assert up["n"].tolist() == [6, 5, 4, 3]
    for got, want in zip(up["s"], [F(5, 6), F(2, 3), F(1, 2), F(1, 3)]):
        assert got == pytest.approx(float(want), abs=1e-15)
    assert K.quantile(up, 0.5) == 5.0


def test_bounds_fasst_kaplan_meier_und_beide_schranken_zusammen(hand):
    t, e = hand
    b = A.bounds(t, e)
    assert (b.km.median, b.lower.median, b.upper.median) == (5.0, 4.0, 5.0)
    assert (b.km.status, b.lower.status, b.upper.status) == (A.STATUS_OK,) * 3


def test_kaplan_meier_liegt_zwischen_den_schranken_im_handbeispiel(hand):
    t, e = hand
    b = A.bounds(t, e)
    for x in (0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0):
        s_lo, s_km, s_up = (A.survival_values(c.km, x) for c in (b.lower, b.km, b.upper))
        assert s_lo <= s_km + 1e-15 and s_km <= s_up + 1e-15, x


def test_gewichtete_kurve_handbeispiel_phi_halb(hand):
    t, e = hand
    time, ev, w = A.phi_split(t, e, 0.5)
    assert len(time) == 8 and w.tolist() == [1, 0.5, 1, 1, 0.5, 1, 0.5, 0.5]
    km = A.weighted_km(time, ev, w)
    assert km["t"].tolist() == [2, 3, 4, 5, 8]
    for got, want in zip(km["s"], [F(5, 6), F(3, 4), F(9, 16), F(9, 32), F(0)]):
        assert got == pytest.approx(float(want), abs=1e-15)
    assert A.median_given_phi(t, e, 0.5) == 5.0


def test_phi_null_ist_kaplan_meier_und_phi_eins_ist_untere_schranke(hand):
    t, e = hand
    km, lo = K.kaplan_meier(t, e), A.lower_bound_km(t, e)
    for phi, ref in ((0.0, km), (1.0, lo)):
        time, ev, w = A.phi_split(t, e, phi)
        wk = A.weighted_km(time, ev, w)
        assert wk["t"].tolist() == ref["t"].tolist()
        assert np.allclose(wk["s"], ref["s"], atol=1e-15)


def test_kipppunkt_handbeispiel(hand):
    """S(4; φ) = (5 − φ)/8 von Hand (Faktoren 5/6, (5 − φ)/5, 3/4): der Median fällt von 5 auf 4 erst bei φ = 1, also kippt „Median ≥ 4,5“ erst dort."""
    t, e = hand
    tip = A.tipping_point(t, e, 4.5)
    assert tip["status"] == A.TIP_TIPS and tip["phi_star"] == pytest.approx(1.0)
    assert tip["phis"].size == 101 and tip["medians"][0] == 5.0 and tip["medians"][-1] == 4.0 and np.all(tip["medians"][:-1] == 5.0)


def test_kipppunkt_drei_zweige(hand):
    t, e = hand
    already = A.tipping_point(t, e, 5.5)       # Kaplan–Meier-Median 5 liegt schon unter 5,5
    assert already["status"] == A.TIP_ALREADY and already["phi_star"] == 0.0
    never = A.tipping_point(t, e, 3.9)         # selbst die untere Schranke (4) liegt darüber
    assert never["status"] == A.TIP_NEVER and math.isnan(never["phi_star"])


def test_median_status_und_nicht_schaetzbar():
    ok = A.make_curve(K.kaplan_meier([1, 2, 3], [1, 1, 1]))
    assert (ok.median, ok.status) == (2.0, A.STATUS_OK)
    no_events = A.make_curve(K.kaplan_meier([1, 2, 3], [0, 0, 0]))
    assert math.isnan(no_events.median) and no_events.status == A.STATUS_NO_EVENTS
    short = A.make_curve(K.kaplan_meier([1, 2, 3, 4], [1, 0, 0, 0]))
    assert math.isnan(short.median) and short.status == A.STATUS_NOT_ESTIMABLE


def test_inside_behandelt_nicht_schaetzbare_obere_schranke_als_unendlich():
    assert A.inside(3.0, 5.0, 4.0) and A.inside(3.0, math.nan, 99.0)
    assert not A.inside(3.0, 5.0, 5.5) and not A.inside(3.0, math.nan, 2.0)
    assert A.inside(3.0, 5.0, 3.0) and A.inside(3.0, 5.0, 5.0)      # Grenzen inklusive


def test_wahre_median_der_stichprobe_ist_ordnungsstatistik():
    """Kleinste Zeit, bei der höchstens die Hälfte länger bleibt: bei n = 6 die dritte, bei n = 5 ebenfalls die dritte, bei n = 4 die zweite."""
    assert A.sample_median_true([5, 1, 3, 9, 7, 2]) == 3.0
    assert A.sample_median_true([5, 1, 3, 9, 7]) == 5.0
    assert A.sample_median_true([4, 1, 3, 2]) == 2.0
