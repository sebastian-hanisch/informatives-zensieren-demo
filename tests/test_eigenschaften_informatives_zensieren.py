"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für jede Instanz gelten müssen, ohne dass man das richtige Ergebnis kennt.

Die Plan-Eigenschaft „Zensierung hinter dem letzten Ereignis ändert die Kurve nicht“ ist falsch (eine zusätzliche Zensierung dort hebt die Kurve) und steht hier nicht."""

import math

import numpy as np
import pytest

import inz_algorithm as A
import inz_constants as C
import inz_core as K
import inz_scenario as S


def _instances(count, seed):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(2, 30))
        t = rng.integers(1, 12, n).astype(float)           # Gleichstände
        e = rng.random(n) < rng.uniform(0.2, 0.9)
        yield t, e


def test_kaplan_meier_liegt_zwischen_den_schranken_an_jeder_stelle():
    grid = np.arange(0.0, 14.0, 0.25)
    for t, e in _instances(300, 1):
        b = A.bounds(t, e)
        lo, km, up = (A.survival_values(c.km, grid) for c in (b.lower, b.km, b.upper))
        assert np.all(lo <= km + 1e-12) and np.all(km <= up + 1e-12)


def test_mediane_sind_geordnet_untere_kleiner_gleich_kaplan_meier_kleiner_gleich_obere():
    for t, e in _instances(300, 2):
        b = A.bounds(t, e)
        assert b.lower.median <= b.km.median + 1e-12 or math.isnan(b.km.median)
        if not math.isnan(b.km.median):
            assert b.upper.median >= b.km.median - 1e-12 or math.isnan(b.upper.median)
        if math.isnan(b.km.median):
            assert math.isnan(b.upper.median)           # erreicht schon Kaplan–Meier 0,5 nicht, dann die obere Schranke erst recht nicht


def test_ohne_ausscheiden_fallen_alle_drei_zusammen():
    for t, _ in _instances(100, 3):
        b = A.bounds(t, np.ones(t.size, dtype=bool))
        for c in (b.lower, b.upper):
            assert np.array_equal(c.km["t"], b.km.km["t"]) and np.allclose(c.km["s"], b.km.km["s"], atol=1e-15)
            assert c.median == b.km.median


def test_obere_schranke_hat_genau_dann_einen_median_wenn_mindestens_die_haelfte_eine_abholung_zeigt():
    for t, e in _instances(300, 4):
        up = A.bounds(t, e).upper
        assert (up.status == A.STATUS_OK) == (2 * int(e.sum()) >= t.size)


def test_untere_schranke_hat_immer_einen_median():
    for t, e in _instances(100, 5):
        assert A.bounds(t, e).lower.status == A.STATUS_OK


def test_zeitachse_skalieren_skaliert_die_mediane_und_laesst_den_kipppunkt_unveraendert():
    for t, e in _instances(60, 6):
        b1, b2 = A.bounds(t, e), A.bounds(3.0 * t, e)
        for c1, c2 in ((b1.km, b2.km), (b1.lower, b2.lower), (b1.upper, b2.upper)):
            assert (math.isnan(c1.median) and math.isnan(c2.median)) or c2.median == pytest.approx(3.0 * c1.median)
        t1, t2 = A.tipping_point(t, e, 4.5, grid=11), A.tipping_point(3.0 * t, e, 13.5, grid=11)
        assert t1["status"] == t2["status"]
        assert (math.isnan(t1["phi_star"]) and math.isnan(t2["phi_star"])) or t1["phi_star"] == t2["phi_star"]


def test_reihenfolge_der_container_ist_gleichgueltig():
    rng = np.random.default_rng(7)
    for t, e in _instances(60, 8):
        perm = rng.permutation(t.size)
        a, b = A.bounds(t, e), A.bounds(t[perm], e[perm])
        for c1, c2 in ((a.km, b.km), (a.lower, b.lower), (a.upper, b.upper)):
            assert np.array_equal(c1.km["s"], c2.km["s"]) and (c1.median == c2.median or (math.isnan(c1.median) and math.isnan(c2.median)))


def test_median_faellt_mit_phi_nicht_ab_phi_eins_ist_die_untere_schranke():
    for t, e in _instances(120, 9):
        meds = np.array([A.median_given_phi(t, e, p) for p in np.linspace(0.0, 1.0, 11)])
        filled = np.where(np.isnan(meds), 1e9, meds)
        assert np.all(np.diff(filled) <= 1e-12)                          # nicht schätzbar zuerst, dann fallend
        assert meds[-1] == A.bounds(t, e).lower.median


def test_mehr_ausscheidende_weiten_die_schranken_bei_gleichen_verweilzeiten():
    """Gleicher Zufalls-Seed: Bei größerem f scheiden mehr Container aus (die Mengen sind ineinander verschachtelt), und jeder Ausgeschiedene hat eine frühere Zeit.
    Dann sinkt die untere Schranke nie, und die obere steigt nie."""
    for seed in range(30):
        prev_lo, prev_up = math.inf, -math.inf
        for f in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8):
            s = S.simulate(C.MECH_NEAR, f, 0.7, 120, seed)
            b = A.bounds(s.obs, s.event)
            up = math.inf if math.isnan(b.upper.median) else b.upper.median
            assert b.lower.median <= prev_lo + 1e-12 and up >= prev_up - 1e-12
            prev_lo, prev_up = b.lower.median, up


def test_wahre_zeiten_der_stichprobe_liegen_in_den_schranken_fuer_beide_mechanismen():
    for mech in C.MECHANISMS:
        for seed in range(40):
            for f, q in ((0.2, 0.7), (0.5, 0.3), (0.9, 0.9)):
                s = S.simulate(mech, f, q, 80, seed)
                b = A.bounds(s.obs, s.event)
                truth = A.sample_median_true(s.true_time)
                assert A.inside(b.lower.median, b.upper.median, truth), (mech, f, q, seed)
                grid = np.linspace(0.0, 30.0, 121)
                s_true = np.array([np.mean(s.true_time > x) for x in grid])
                assert np.all(A.survival_values(b.lower.km, grid) <= s_true + 1e-12) and np.all(s_true <= A.survival_values(b.upper.km, grid) + 1e-12)


def test_kaplan_meier_aus_dem_kern_aendert_sich_nicht_bei_umstellen_der_zensierten_hinter_die_letzte_zeit_ausser_obere_schranke():
    """Die obere Schranke ist Kaplan–Meier mit allen Zensierten hinter der letzten Zeit; setzt man sie schon bei Kaplan–Meier dorthin (statt an ihre Zeit), ist das genau die obere Schranke."""
    for t, e in _instances(100, 10):
        late = np.where(e, t, t.max() + 1.0)
        km_late = K.kaplan_meier(late, e)
        up = A.upper_bound_km(t, e)
        assert np.array_equal(km_late["t"], up["t"]) and np.array_equal(km_late["s"], up["s"])
