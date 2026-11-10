"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, alle zensiert, f = 0 und f = 1, Grenzen inklusive,
ungültige Eingaben (verständliche Fehler statt stiller Zahlen) und die Status der Schranken."""

import math

import numpy as np
import pytest

import inz_algorithm as A
import inz_constants as C
import inz_core as K
import inz_ersatz as E
import inz_evaluation as ev
import inz_scenario as S


def test_ein_container_beobachtet_und_ausgeschieden():
    seen = A.bounds([3.0], [True])
    assert (seen.km.median, seen.lower.median, seen.upper.median) == (3.0, 3.0, 3.0)
    gone = A.bounds([3.0], [False])
    assert math.isnan(gone.km.median) and gone.km.status == A.STATUS_NO_EVENTS
    assert gone.lower.median == 3.0 and math.isnan(gone.upper.median) and gone.upper.status == A.STATUS_NO_EVENTS


def test_genau_zwei_container():
    """n = 2: ein Ereignis und eine Ausscheidung. Median der unteren Schranke: die kleinere Zeit (S fällt bei ihr auf 1/2); obere Schranke: S = 1/2 nach dem Ereignis, also Median = Ereigniszeit."""
    b = A.bounds([2.0, 5.0], [True, False])
    assert b.lower.median == 2.0 and b.km.median == 2.0 and b.upper.median == 2.0
    b = A.bounds([2.0, 5.0], [False, True])
    assert b.lower.median == 2.0 and b.km.median == 5.0 and b.upper.median == 5.0      # Kaplan–Meier: n = 1 bei t = 5, S = 0
    assert A.sample_median_true([2.0, 5.0]) == 2.0


def test_alle_ausgeschieden_obere_schranke_unendlich_untere_definiert():
    b = A.bounds([1.0, 2.0, 2.0, 7.0], [False] * 4)
    assert b.lower.median == 2.0 and math.isnan(b.upper.median) and math.isnan(b.km.median)
    assert A.inside(b.lower.median, b.upper.median, 100.0)


def test_alle_gleichzeitig_gleiche_zeit_ereignis_und_ausscheiden():
    t, e = [4.0, 4.0, 4.0, 4.0], [True, False, True, False]
    b = A.bounds(t, e)
    assert b.km.median == 4.0                      # Ereignisse zuerst: 2 von 4 gehen bei t = 4 ab, S = 1/2
    assert b.lower.median == 4.0 and b.upper.median == 4.0


def test_f_null_und_f_eins_in_der_auswertung():
    a0 = ev.analyse(ev.Settings(C.MECH_NEAR, 0, 70, 100, 3))
    assert a0.sample.n_events == 100 and a0.bounds.km.median == a0.bounds.lower.median == a0.bounds.upper.median == a0.sample_truth
    a1 = ev.analyse(ev.Settings(C.MECH_NEAR, 100, 70, 100, 3))
    assert a1.sample.n_events == 0 and math.isnan(a1.bounds.km.median) and math.isnan(a1.bounds.upper.median) and a1.bounds.lower.status == A.STATUS_OK
    kind, text = ev.verdict_core(a1)
    assert kind == "warning" and "keine einzige Abholung" in text


def test_grenzen_der_regler_sind_inklusive():
    for mech in C.MECHANISMS:
        for f, q, n in ((C.F_MIN, C.Q_MIN, C.N_MIN), (C.F_MAX, C.Q_MAX, C.N_MAX), (C.F_MIN, C.Q_MAX, C.N_MIN), (C.F_MAX, C.Q_MIN, C.N_MAX)):
            a = ev.analyse(ev.Settings(mech, f, q, n, 11))
            assert a.sample.n == n and a.bounds.lower.status == A.STATUS_OK
            assert A.inside(a.bounds.lower.median, a.bounds.upper.median, a.sample_truth)
            ev.verdict_core(a)                    # wirft nichts, auch an den Rändern


def test_ungueltige_eingaben_geben_verstaendliche_fehler():
    for bad in (([0.0, 1.0], [True, True]), ([1.0, -2.0], [True, True]), ([1.0, np.inf], [True, True]), ([1.0, 2.0], [True]), ([], [])):
        with pytest.raises(K.SurvivalError):
            A.bounds(*bad)
    with pytest.raises(K.SurvivalError):
        A.phi_split([1.0, 2.0], [True, False], 1.5)
    with pytest.raises(K.SurvivalError):
        A.weighted_km([1.0, 2.0], [True, True], [1.0, -1.0])
    with pytest.raises(K.SurvivalError):
        A.weighted_km([1.0, 2.0], [True, True], [1.0])
    with pytest.raises(ValueError):
        A.tipping_point([1.0, 2.0], [True, False], 1.0, grid=1)


def test_gewichtete_kurve_ohne_ereignis_und_mit_null_gewichten():
    km = A.weighted_km([1.0, 2.0], [False, False], [1.0, 1.0])
    assert km["t"].size == 0 and km["s"].size == 0
    km = A.weighted_km([1.0, 2.0, 3.0], [True, True, True], [0.0, 1.0, 1.0])           # Gewicht 0: die Zeile zählt nicht
    assert km["t"].tolist() == [2.0, 3.0] and km["s"][0] == pytest.approx(0.5)


def test_ersatzwelt_randfaelle():
    for mech in C.MECHANISMS:
        none = E.ersatz_world(mech, 0.0, 0.7)
        assert np.allclose(none.s_b, none.s_true, atol=1e-4)          # niemand scheidet aus: Welt B ist Welt A
        every = E.ersatz_world(mech, 1.0, 0.7)
        assert np.all(every.s_b == 1.0)                                # keine Abholung gesehen: die Ersatzwelt hat nie ein Ereignis
        sb = E.sample_world_b(every, 50, 1)
        assert not sb.event.any() and np.all(sb.obs > 0)
    with pytest.raises(ValueError):
        E.ersatz_world(C.MECH_NEAR, 0.4, 1.0)
    with pytest.raises(ValueError):
        E.sample_world_b(E.ersatz_world(C.MECH_NEAR, 0.4, 0.7), 0, 1)


def test_raster_randfaelle():
    assert np.allclose(E.raster_survival(C.MECH_NEAR, 1.0, 0.7, 1.0, [4, 6, 8]), 1.0)       # f = 1: keine Abholung gesehen, S bleibt 1
    s0 = E.raster_survival(C.MECH_NEAR, 0.0, 0.7, 1.0, [4, 6, 8], "event_first")
    assert np.allclose(s0, E.raster_survival(C.MECH_NEAR, 0.0, 0.7, 1.0, [4, 6, 8], "censor_first"))   # ohne Ausscheiden gibt es nichts, dessen Regel zählt
    with pytest.raises(ValueError):
        E.raster_survival(C.MECH_NEAR, 0.4, 0.7, 0.0, [4])
    with pytest.raises(ValueError):
        E.raster_survival(C.MECH_NEAR, 0.4, 0.7, 1.0, [4], "egal")


def test_empirische_anteile_sind_nicht_fallend_und_summieren_hoechstens_eins():
    s = S.simulate(C.MECH_NEAR, 0.4, 0.7, 500, 2)
    grid = np.linspace(0.0, 60.0, 61)
    e, c = E.empirical_cifs(s, grid)
    assert np.all(np.diff(e) >= 0) and np.all(np.diff(c) >= 0) and np.all(e + c <= 1.0 + 1e-12)
    assert e[-1] + c[-1] == pytest.approx(1.0, abs=1e-12)      # nach 60 Tagen ist alles entschieden (D > 60 hat Wahrscheinlichkeit unter 1e-6)
