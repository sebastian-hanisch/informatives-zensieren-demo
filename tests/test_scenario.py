"""Szenario: Erzeuger der Stichprobe (Mechanismen, Zufalls-Seed, Parameter) gegen ihre Definition."""

import math

import numpy as np
import pytest

import inz_constants as C
import inz_scenario as S


def test_gleicher_zufalls_seed_gleiche_stichprobe_anderer_seed_andere():
    a = S.simulate(C.MECH_NEAR, 0.4, 0.7, 50, 3)
    b = S.simulate(C.MECH_NEAR, 0.4, 0.7, 50, 3)
    c = S.simulate(C.MECH_NEAR, 0.4, 0.7, 50, 4)
    assert np.array_equal(a.obs, b.obs) and np.array_equal(a.event, b.event) and np.array_equal(a.true_time, b.true_time)
    assert not np.array_equal(a.obs, c.obs)


def test_verweilzeiten_sind_in_beiden_mechanismen_dieselben():
    a = S.simulate(C.MECH_NEAR, 0.4, 0.7, 200, 9)
    b = S.simulate(C.MECH_LONG, 0.4, 0.7, 200, 9)
    assert np.array_equal(a.true_time, b.true_time)


def test_beobachtet_wird_q_mal_d_fuer_ausgeschiedene_sonst_d():
    for mech in C.MECHANISMS:
        s = S.simulate(mech, 0.5, 0.3, 500, 21)
        assert np.allclose(s.obs[s.event], s.true_time[s.event])
        assert np.allclose(s.obs[~s.event], 0.3 * s.true_time[~s.event])
        assert np.all(s.obs <= s.true_time)           # Ausscheiden liegt vor der Abholung (q < 1)


def test_langlieger_scheiden_genau_oberhalb_der_schwelle_aus():
    s = S.simulate(C.MECH_LONG, 0.4, 0.7, 5000, 5)
    d = S.cut_threshold(0.4)
    assert np.all(s.true_time[~s.event] > d) and np.all(s.true_time[s.event] <= d)
    assert abs(s.censored_share - 0.4) < 4.5 * math.sqrt(0.4 * 0.6 / 5000)


def test_kurz_vor_der_abholung_trifft_zufaellig_und_unabhaengig_von_d():
    n = 40_000
    s = S.simulate(C.MECH_NEAR, 0.3, 0.7, n, 6)
    assert abs(s.censored_share - 0.3) < 4.5 * math.sqrt(0.3 * 0.7 / n)
    # der Anteil der Ausgeschiedenen unter den kurzen und unter den langen Verweilzeiten ist derselbe (Unabhängigkeit des Ausscheidens von D bei diesem Mechanismus)
    short = s.true_time < 4.0
    for part in (short, ~short):
        assert abs((~s.event[part]).mean() - 0.3) < 4.5 * math.sqrt(0.3 * 0.7 / part.sum())


def test_rand_f_null_und_eins():
    for mech in C.MECHANISMS:
        none = S.simulate(mech, 0.0, 0.7, 100, 1)
        assert none.event.all() and np.array_equal(none.obs, none.true_time)
        every = S.simulate(mech, 1.0, 0.7, 100, 1)
        assert not every.event.any() and np.allclose(every.obs, 0.7 * every.true_time)


def test_schwelle_der_langlieger_randfaelle():
    assert S.cut_threshold(0.0) == math.inf and S.cut_threshold(1.0) == 0.0
    assert S.cut_threshold(0.5) == pytest.approx(4.0)            # die Hälfte bleibt länger als der Median


def test_ungueltige_parameter_werden_abgelehnt():
    with pytest.raises(ValueError):
        S.simulate("unbekannt", 0.4, 0.7, 10, 1)
    for f, q, n in ((-0.1, 0.7, 10), (1.1, 0.7, 10), (0.4, 0.0, 10), (0.4, 1.0, 10), (0.4, 0.7, 0)):
        with pytest.raises(ValueError):
            S.simulate(C.MECH_NEAR, f, q, n, 1)


def test_wahrheit_median_vier_und_mittel():
    assert float(S.true_survival(4.0)) == pytest.approx(0.5)
    assert S.true_quantile(0.5) == pytest.approx(4.0)
    assert S.true_mean() == pytest.approx(4.0 * math.exp(0.9**2 / 2))
    assert float(S.true_survival(0.0)) == 1.0 and float(S.true_density(0.0)) == 0.0


def test_stichprobenmedian_streut_um_den_wahren_median():
    meds = [float(np.median(S.simulate(C.MECH_NEAR, 0.0, 0.7, 300, 100 + i).true_time)) for i in range(60)]
    assert abs(np.mean(meds) - 4.0) < 0.1
