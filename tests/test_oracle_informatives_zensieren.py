"""Orakel-Tests (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen ANDEREN Rechenweg.

  Schranken            Bruch-Zählung (Anteil der Zeiten > t) und Aufzählung aller möglichen wahren Zeiten der Ausgeschiedenen (beide Extreme werden angenommen)
  gewichtete Kurve     Bruchrechnung mit Gewichten und Vervielfältigung der Zeilen durch den ungewichteten Kern
  Kipppunkt            Raster der Mediane über die Vervielfältigung (ganzzahlige Anteile), unabhängig von der gewichteten Kurve
  Wahrheit             scipy.stats.lognorm (Verteilungsfunktion, Dichte, Quantil, Mittel)
  Teilverteilungen     Simulation mit SplitMix64 (Band über viele Container) und numerische Quadratur der Teildichten (scipy.integrate.quad)
  Ersatzwelt           Quadratur der Hazards, S_B · S_C = S_obs, Stichproben beider Welten (gleiche beobachtbare Anteile, Kaplan–Meier gegen die geschlossene Form)
                       und die exakte Bruchrechnung der diskreten Fassung (zwei Welten, gleiche beobachtbare Wahrscheinlichkeiten)
  Tagesraster          Kaplan–Meier auf aufgerundeten Zeiten großer Stichproben gegen die Formel ohne Zufall
Zufallsinstanzen mit numpy (nur für Eigenschaften, die für jede Instanz gelten; keine Claim-Zahl hängt daran)."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

import inz_algorithm as A
import inz_constants as C
import inz_core as K
import inz_ersatz as E
import inz_scenario as S

scipy_stats = pytest.importorskip("scipy.stats")
scipy_integrate = pytest.importorskip("scipy.integrate")


def _random_sample(rng, n):
    t = rng.integers(1, 10, n).astype(float)           # ganzzahlig: viele Gleichstände
    e = rng.random(n) < 0.6
    return t, e


# ------------------------------------------------------------------------------------------------ Schranken


def _median_from_fractions(surv, times):
    """Kleinste Zeit aus `times` mit S <= 1/2 (Brüche); nan, wenn es keine gibt."""
    for x in sorted(set(times)):
        if surv(x) <= F(1, 2):
            return float(x)
    return math.nan


def test_schranken_gegen_bruchzaehlung_auf_zufallsinstanzen():
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(1, 14))
        t, e = _random_sample(rng, n)
        ti = [int(x) for x in t]
        ev = [bool(x) for x in e]
        lo_S = lambda x: F(sum(1 for v in ti if v > x), n)                                  # noqa: E731  alle Zeiten sind Ereignisse
        up_S = lambda x: 1 - F(sum(1 for v, g in zip(ti, ev) if g and v <= x), n)           # noqa: E731  Ausgeschiedene bleiben für immer
        b = A.bounds(t, e)
        for x in range(0, 11):
            assert A.survival_values(b.lower.km, x) == pytest.approx(float(lo_S(x)), abs=1e-12)
            assert A.survival_values(b.upper.km, x) == pytest.approx(float(up_S(x)), abs=1e-12)
        want_lo = _median_from_fractions(lo_S, ti)
        want_up = _median_from_fractions(up_S, [v for v, g in zip(ti, ev) if g])
        assert b.lower.median == want_lo or (math.isnan(b.lower.median) and math.isnan(want_lo))
        assert (b.upper.median == want_up) or (math.isnan(b.upper.median) and math.isnan(want_up)), (ti, ev)


def test_schranken_gegen_aufzaehlung_aller_moeglichen_wahren_zeiten():
    """Kleine Stichproben: Jeder Ausgeschiedene bekommt jede mögliche wahre Zeit (eine beobachtete Zeit ab seiner eigenen oder „nie“). Der Median der wahren Zeiten
    liegt in jeder Belegung zwischen den Schranken, und beide Extreme werden angenommen."""
    rng = np.random.default_rng(12)
    never = 10**6
    checked = 0
    for _ in range(60):
        n = int(rng.integers(2, 7))
        t, e = _random_sample(rng, n)
        if e.all() or n < 2:
            continue
        ti = [int(x) for x in t]
        cens = [i for i in range(n) if not e[i]]
        options = [sorted({v for v in ti if v >= ti[i]} | {never}) for i in cens]
        k = math.ceil(n / 2)
        medians = []
        for combo in itertools.product(*options):
            true = list(ti)
            for i, v in zip(cens, combo):
                true[i] = v
            medians.append(sorted(true)[k - 1])
        b = A.bounds(t, e)
        lo = b.lower.median
        up = math.inf if math.isnan(b.upper.median) else b.upper.median
        assert min(medians) == lo
        assert (max(medians) >= never) if math.isinf(up) else (max(medians) == up)
        assert all(lo <= m <= (never if math.isinf(up) else up) for m in medians)
        checked += 1
    assert checked >= 40


# ------------------------------------------------------------------------------------------------ gewichtete Kurve und Kipppunkt


def _km_fractions(times, events, weights):
    """Kaplan–Meier mit Gewichten in Brüchen (Schleife über die Ereigniszeiten); Rückgabe {t: S(t)}."""
    s, out = F(1), {}
    for t in sorted({x for x, g, w in zip(times, events, weights) if g and w > 0}):
        n = sum((w for x, w in zip(times, weights) if x >= t), F(0))
        d = sum((w for x, g, w in zip(times, events, weights) if g and x == t), F(0))
        s *= (n - d) / n
        out[t] = s
    return out


def test_gewichtete_kurve_gegen_bruchrechnung():
    rng = np.random.default_rng(13)
    for _ in range(120):
        n = int(rng.integers(2, 12))
        t, e = _random_sample(rng, n)
        phi = F(int(rng.integers(0, 5)), 4)
        time, ev, w = A.phi_split(t, e, float(phi))
        km = A.weighted_km(time, ev, w)
        times = [int(x) for x in time]
        weights = [F(x).limit_denominator(1000) for x in w]
        want = _km_fractions(times, [bool(x) for x in ev], weights)
        assert km["t"].tolist() == sorted(want)
        for got, ref in zip(km["s"], [want[k] for k in sorted(want)]):
            assert got == pytest.approx(float(ref), abs=1e-12)


def _replicated_median(t, e, k, m):
    """φ = k/m über ganze Zeilen: jede Zeile m-mal; jeder Ausgeschiedene k-mal als Ereignis und (m − k)-mal als Zensierter. Ungewichteter Kern."""
    tt, ee = [], []
    for x, g in zip(t, e):
        if g:
            tt += [x] * m
            ee += [True] * m
        else:
            tt += [x] * m
            ee += [True] * k + [False] * (m - k)
    return K.quantile(K.kaplan_meier(tt, ee), 0.5)


def test_median_zu_phi_gegen_vervielfaeltigung_der_zeilen():
    rng = np.random.default_rng(14)
    for _ in range(120):
        n = int(rng.integers(2, 12))
        t, e = _random_sample(rng, n)
        k = int(rng.integers(0, 5))
        want = _replicated_median(t, e, k, 4)
        got = A.median_given_phi(t, e, k / 4)
        assert (math.isnan(got) and math.isnan(want)) or got == want, (t.tolist(), e.tolist(), k)


def test_kipppunkt_gegen_raster_ueber_vervielfaeltigung():
    """Raster 0, 1/4, 1/2, 3/4, 1: kleinstes φ, bei dem der Median unter die Schwelle fällt (nicht schätzbar zählt als „gilt“), unabhängig von der gewichteten Kurve gerechnet."""
    rng = np.random.default_rng(15)
    for _ in range(100):
        n = int(rng.integers(2, 12))
        t, e = _random_sample(rng, n)
        thr = float(rng.integers(2, 9)) + 0.5
        meds = [_replicated_median(t, e, k, 4) for k in range(5)]
        holds = [math.isnan(m) or m >= thr for m in meds]
        tip = A.tipping_point(t, e, thr, grid=5)
        if not holds[0]:
            assert tip["status"] == A.TIP_ALREADY
        elif all(holds):
            assert tip["status"] == A.TIP_NEVER
        else:
            assert tip["status"] == A.TIP_TIPS and tip["phi_star"] == pytest.approx(holds.index(False) / 4)


# ------------------------------------------------------------------------------------------------ Wahrheit und Teilverteilungen


def test_wahrheit_gegen_scipy_lognorm():
    dist = scipy_stats.lognorm(s=C.SIGMA_TRUE, scale=C.MEDIAN_TRUE)
    t = np.array([0.0, 0.3, 1.0, 2.0, 4.0, 6.0, 8.0, 13.0, 40.0])
    assert np.allclose(S.true_survival(t), dist.sf(t), atol=1e-14)
    assert np.allclose(S.true_density(t), dist.pdf(t), atol=1e-14)
    assert S.true_quantile(0.5) == pytest.approx(4.0)
    for p in (0.1, 0.5, 0.9):
        assert S.true_quantile(p) == pytest.approx(float(dist.ppf(p)), rel=1e-12)
    assert S.true_mean() == pytest.approx(float(dist.mean()), rel=1e-12)
    for f in (0.1, 0.4, 0.5, 0.9):
        assert float(S.true_survival(S.cut_threshold(f))) == pytest.approx(f, abs=1e-12)


@pytest.mark.parametrize("mech, f, q", [(C.MECH_NEAR, 0.4, 0.7), (C.MECH_NEAR, 0.2, 0.3), (C.MECH_LONG, 0.5, 0.3), (C.MECH_LONG, 0.4, 0.7)])
def test_teilverteilungen_gegen_simulation_und_quadratur(mech, f, q):
    n = 200_000
    s = S.simulate(mech, f, q, n, 777)
    for x in (1.5, 3.0, 4.0, 6.0, 10.0):
        for emp, closed in (
            (np.mean(s.obs > x), float(E.s_obs(mech, f, q, x))),
            (np.mean(s.event & (s.obs <= x)), float(E.cif_event(mech, f, q, x))),
            (np.mean(~s.event & (s.obs <= x)), float(E.cif_cens(mech, f, q, x))),
        ):
            assert abs(emp - closed) < 4.5 * math.sqrt(max(closed * (1 - closed), 1e-4) / n), (x, emp, closed)
        # Quadratur der Teildichten gegen die Verteilungsfunktionen (andere Methode als die geschlossene Formel)
        qe = scipy_integrate.quad(lambda u: float(E.q_event(mech, f, q, u)), 0, x, limit=200, points=[S.cut_threshold(f)] if 0 < S.cut_threshold(f) < x else None)[0]
        qc = scipy_integrate.quad(lambda u: float(E.q_cens(mech, f, q, u)), 0, x, limit=200, points=[q * S.cut_threshold(f)] if 0 < q * S.cut_threshold(f) < x else None)[0]
        assert qe == pytest.approx(float(E.cif_event(mech, f, q, x)), abs=1e-7)
        assert qc == pytest.approx(float(E.cif_cens(mech, f, q, x)), abs=1e-7)
        assert float(E.s_obs(mech, f, q, x)) == pytest.approx(1.0 - qe - qc, abs=1e-7)


# ------------------------------------------------------------------------------------------------ Ersatzwelt


@pytest.mark.parametrize("mech, f, q", [(C.MECH_NEAR, 0.4, 0.7), (C.MECH_NEAR, 0.2, 0.5), (C.MECH_LONG, 0.5, 0.3)])
def test_ersatzwelt_gegen_quadratur_der_hazards(mech, f, q):
    w = E.ersatz_world(mech, f, q)
    for x in (2.0, 4.0, 6.0, 8.0):
        h_e = scipy_integrate.quad(lambda u: float(E.q_event(mech, f, q, u)) / float(E.s_obs(mech, f, q, u)), 0, x, limit=400, epsabs=1e-12,
                                   points=[S.cut_threshold(f)] if 0 < S.cut_threshold(f) < x else None)[0]
        assert w.at("s_b", x) == pytest.approx(math.exp(-h_e), abs=2e-5)
    assert float(np.max(np.abs(w.s_b * w.s_c - w.s_obs))) < 2e-5       # S_B · S_C = S_obs: die Ersatzwelt hat die beobachtbare Überlebensfunktion
    assert w.s_true[0] == 1.0 and w.s_b[0] == 1.0 and np.all(np.diff(w.s_b) <= 1e-12)


def _km_at(obs, ev, xs):
    """Kaplan–Meier an den Stellen xs für stetige Zeiten ohne Gleichstände (schnelle Fassung: n_i = n − i), unabhängig vom Kern."""
    order = np.argsort(obs, kind="stable")
    e = ev[order]
    n = obs.size
    surv = np.cumprod(1.0 - np.where(e, 1.0 / (n - np.arange(n)), 0.0))
    idx = np.searchsorted(obs[order], xs, side="right") - 1
    return np.where(idx < 0, 1.0, surv[np.maximum(idx, 0)])


def test_schnelle_kaplan_meier_fassung_stimmt_mit_dem_kern_ueberein():
    s = S.simulate(C.MECH_NEAR, 0.4, 0.7, 300, 5)
    xs = np.array([1.0, 3.0, 4.0, 6.0, 9.0, 15.0])
    km = K.kaplan_meier(s.obs, s.event)
    assert np.allclose(_km_at(s.obs, s.event, xs), [K.survival_at(km, x) for x in xs], atol=1e-12)


@pytest.mark.parametrize("mech, f, q", [(C.MECH_NEAR, 0.4, 0.7), (C.MECH_LONG, 0.5, 0.3)])
def test_zwei_welten_gleiche_beobachtbare_anteile_verschiedene_wahrheit(mech, f, q):
    w = E.ersatz_world(mech, f, q)
    n = 100_000
    sa = S.simulate(mech, f, q, n, 4242)
    sb = E.sample_world_b(w, n, 4243)
    grid = np.linspace(0.0, 30.0, 301)
    ea, ca = E.empirical_cifs(sa, grid)
    eb, cb = E.empirical_cifs(sb, grid)
    assert max(np.abs(ea - eb).max(), np.abs(ca - cb).max()) < 0.012       # KS-Größenordnung 1,36·√(2/n) = 0,006
    xs = np.array([4.0, 6.0, 8.0])
    for sample in (sa, sb):
        km = _km_at(sample.obs, sample.event, xs)
        assert np.allclose(km, [w.at("s_b", x) for x in xs], atol=0.008)       # Kaplan–Meier schätzt die Ersatzwelt ...
    wahr = [w.at("s_true", x) for x in xs]
    assert max(abs(a - b) for a, b in zip(wahr, [w.at("s_b", x) for x in xs])) > 0.1     # ... und die wahre Kurve liegt weit davon entfernt


def _observable(joint):
    """Beobachtbare Wahrscheinlichkeiten aus der gemeinsamen Verteilung {(D, C): p} (C = None: nie). Ereignis zuerst bei Gleichstand: D <= C zählt als gesehene Abholung."""
    e, c = {}, {}
    for (d, cc), p in joint.items():
        if cc is None or d <= cc:
            e[d] = e.get(d, F(0)) + p
        else:
            c[cc] = c.get(cc, F(0)) + p
    return e, c


def _ersatz_hazards(e, c, k):
    """Hazards der Ersatzwelt in Brüchen: h_j = e_j / r_j (Abholung), k_j = c_j / (r_j − e_j) (Ausscheiden unter denen, die bei j nicht abgeholt werden)."""
    risk, h, kk = F(1), {}, {}
    for j in range(1, k + 1):
        ej, cj = e.get(j, F(0)), c.get(j, F(0))
        h[j] = ej / risk if risk > 0 else F(0)
        kk[j] = cj / (risk - ej) if risk - ej > 0 else F(0)
        risk -= ej + cj
    return h, kk


def test_diskrete_fassung_in_bruechen_zwei_welten_gleiche_beobachtungen():
    """Winzige Welt (Zeiten 1 bis 4), 40 zufällige abhängige gemeinsame Verteilungen von (D, C) als Brüche. Die Ersatzwelt (D_B und C_B unabhängig, mit den Hazards h und k)
    erzeugt GENAU dieselben beobachtbaren Wahrscheinlichkeiten; ihre Kurve ist S_B = Π(1 − h_j). Bei mindestens der Hälfte der Welten weicht sie von der wahren Kurve P(D > j) ab."""
    rng = np.random.default_rng(16)
    kmax, differ = 4, 0
    for _ in range(40):
        cells = {(d, c): F(int(rng.integers(0, 6))) for d in range(1, kmax + 1) for c in list(range(1, kmax + 1)) + [None]}
        cells[(1, None)] += 1
        total = sum(cells.values())
        joint = {key: v / total for key, v in cells.items()}
        e, c = _observable(joint)
        h, kk = _ersatz_hazards(e, c, kmax)
        prod_h = prod_k = F(1)
        s_b, true_s = F(1), []
        for j in range(1, kmax + 1):
            assert prod_h * prod_k * h[j] == e.get(j, F(0))                      # Abholung gesehen bei j: D_B = j, C_B >= j
            assert prod_h * prod_k * (1 - h[j]) * kk[j] == c.get(j, F(0))        # ausgeschieden bei j: C_B = j, D_B > j
            prod_h *= 1 - h[j]
            prod_k *= 1 - kk[j]
            s_b *= 1 - h[j]
            true_s.append((s_b, sum((p for (d, _), p in joint.items() if d > j), F(0))))
        differ += any(a != b for a, b in true_s)
    assert differ >= 20


def test_diskrete_fassung_unabhaengige_welt_ist_ihre_eigene_ersatzwelt():
    pd = [F(3), F(2), F(1), F(2)]
    pc = [F(1), F(3), F(2), F(1), F(1)]      # Zeiten 1 bis 4 und „nie“
    joint = {(d + 1, (c + 1) if c < 4 else None): (pd[d] / sum(pd)) * (pc[c] / sum(pc)) for d in range(4) for c in range(5)}
    e, c = _observable(joint)
    h, _ = _ersatz_hazards(e, c, 4)
    s_b = F(1)
    for j in range(1, 5):
        s_b *= 1 - h[j]
        assert s_b == sum((p for (d, _), p in joint.items() if d > j), F(0))


def test_diskrete_fassung_von_hand_abhaengige_welt_andere_wahrheit():
    """Von Hand: (D = 1, C = nie) 1/2, (D = 2, C = nie) 1/4, (D = 3, C = 1) 1/4: der Container mit D = 3 scheidet bei 1 aus, bevor er abgeholt würde.
    Beobachtbar: Abholung bei 1 mit 1/2, Ausscheiden bei 1 mit 1/4, Abholung bei 2 mit 1/4. Risiko r1 = 1, r2 = 1/4.
    Hazards: h1 = 1/2, k1 = (1/4)/(1 − 1/2) = 1/2, h2 = (1/4)/(1/4) = 1. Ersatz S_B(1) = 1/2, S_B(2) = 0; wahr P(D > 1) = 1/2, P(D > 2) = 1/4.
    Welt B gibt die Beobachtungen wieder: 1/2, (1 − 1/2) · 1/2 = 1/4 und (1 − 1/2)(1 − 1/2) · 1 = 1/4."""
    joint = {(1, None): F(1, 2), (2, None): F(1, 4), (3, 1): F(1, 4)}
    e, c = _observable(joint)
    assert (e, c) == ({1: F(1, 2), 2: F(1, 4)}, {1: F(1, 4)})
    h, kk = _ersatz_hazards(e, c, 3)
    assert (h[1], kk[1], h[2]) == (F(1, 2), F(1, 2), F(1))
    assert (h[1], (1 - h[1]) * kk[1], (1 - h[1]) * (1 - kk[1]) * h[2]) == (F(1, 2), F(1, 4), F(1, 4))
    assert (1 - h[1], (1 - h[1]) * (1 - h[2])) == (F(1, 2), F(0))
    assert (sum(p for (d, _), p in joint.items() if d > 1), sum(p for (d, _), p in joint.items() if d > 2)) == (F(1, 2), F(1, 4))


# ------------------------------------------------------------------------------------------------ Tagesraster


@pytest.mark.parametrize("width", [1.0, 0.5])
def test_tagesraster_gegen_kaplan_meier_auf_aufgerundeten_zeiten(width):
    n = 200_000
    s = S.simulate(C.MECH_NEAR, 0.4, 0.7, n, 9090)
    k = np.ceil(s.obs / width)
    xs = [4, 6, 8]
    for order, shift in (("event_first", 0.0), ("censor_first", 1e-6)):
        # Ausgeschiedene im selben Fenster scheiden bei „Zensierung zuerst“ vor den Abholungen aus: ihre Zeit liegt knapp davor
        time = np.where(s.event, k * width, k * width - shift * width)
        km = K.kaplan_meier(time, s.event)
        want = E.raster_survival(C.MECH_NEAR, 0.4, 0.7, width, xs, order)
        got = [K.survival_at(km, x) for x in xs]
        assert np.allclose(got, want, atol=0.008), (order, got, want)


def test_tagesraster_naehert_sich_der_stetigen_ersatzwelt():
    w = E.ersatz_world(C.MECH_NEAR, 0.4, 0.7)
    xs = [4, 6, 8]
    cont = np.array([w.at("s_b", x) for x in xs])
    prev = None
    for width in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
        ef = np.array(E.raster_survival(C.MECH_NEAR, 0.4, 0.7, width, xs, "event_first"))
        cf = np.array(E.raster_survival(C.MECH_NEAR, 0.4, 0.7, width, xs, "censor_first"))
        assert np.all(ef > cont) and np.all(cf < cont)               # die zwei Regeln schließen die stetige Kurve ein
        gap = float((ef - cf).max())
        assert prev is None or gap < prev
        prev = gap
    assert prev < 0.01
