"""Die unabhängige Ersatzwelt: dieselben beobachtbaren Daten, aber unabhängige Zeiten (unser Schluss aus Tsiatis 1975, Satz 2; Aussagegrenzen in der README).

Aus den beobachtbaren Teildichten von (Zeit, Ereignis) wird gebaut:
  q_E(t) = Dichte, bei t abgeholt zu werden und dabei beobachtet zu sein,   q_C(t) = Dichte, bei t auszuscheiden,   S_obs(t) = P(beobachtete Zeit > t),
  Hazards h_E = q_E / S_obs und h_C = q_C / S_obs   (die „ursachenspezifischen“ Hazards von Ereignis und Ausscheiden),
  S_B(t) = exp(−∫ h_E)  und  S_C(t) = exp(−∫ h_C)   (Netto-Überlebensfunktionen der Ersatzwelt B: Ereigniszeit und Ausscheidezeit sind dort UNABHÄNGIG).
Dann gilt S_B · S_C = S_obs, und Welt B erzeugt dieselben q_E und q_C wie die abhängige Welt A. Kaplan–Meier schätzt aus den Daten die Kurve S_B, nicht die wahre Kurve S von A.
Das Gitter ist stetig (kein Tagesraster); `raster_survival` rechnet, was ein Tagesraster ändert."""

import math
from dataclasses import dataclass

import numpy as np

import inz_constants as C
import inz_scenario as S
from inz_core import SplitMix64

GRID_START = 1e-3
GRID_POINTS = 6001


def _check(mech: str, f: float, q: float) -> None:
    if mech not in C.MECHANISMS:
        raise ValueError(f"Unbekannter Mechanismus: {mech!r}")
    if not (0.0 <= f <= 1.0) or not (0.0 < q < 1.0):
        raise ValueError("f muss in [0, 1] und q in (0, 1) liegen.")


# ------------------------------------------------------------------------------------------------ Beobachtbare Teilverteilungen in geschlossener Form


def cif_event(mech: str, f: float, q: float, t):
    """P(Ereignis beobachtet und Zeit <= t): Abholung gesehen bis t."""
    _check(mech, f, q)
    t = np.asarray(t, dtype=float)
    if mech == C.MECH_NEAR:
        return (1.0 - f) * (1.0 - S.true_survival(t))
    return 1.0 - S.true_survival(np.minimum(t, S.cut_threshold(f)))


def cif_cens(mech: str, f: float, q: float, t):
    """P(vorzeitig ausgeschieden und Ausscheidezeit <= t)."""
    _check(mech, f, q)
    t = np.asarray(t, dtype=float)
    if mech == C.MECH_NEAR:
        return f * (1.0 - S.true_survival(t / q))
    if f <= 0.0:
        return np.zeros_like(t)
    return np.maximum(f - S.true_survival(t / q), 0.0)


def s_obs(mech: str, f: float, q: float, t):
    """P(beobachtete Zeit > t) = 1 − cif_event − cif_cens."""
    return 1.0 - cif_event(mech, f, q, t) - cif_cens(mech, f, q, t)


def q_event(mech: str, f: float, q: float, t):
    """Teildichte der gesehenen Abholungen."""
    _check(mech, f, q)
    t = np.asarray(t, dtype=float)
    if mech == C.MECH_NEAR:
        return (1.0 - f) * S.true_density(t)
    return np.where(t <= S.cut_threshold(f), S.true_density(t), 0.0)


def q_cens(mech: str, f: float, q: float, t):
    """Teildichte des Ausscheidens (Ausscheidezeit q·D)."""
    _check(mech, f, q)
    t = np.asarray(t, dtype=float)
    dens = S.true_density(t / q) / q
    if mech == C.MECH_NEAR:
        return f * dens
    if f <= 0.0:
        return np.zeros_like(t)
    return np.where(t / q > S.cut_threshold(f), dens, 0.0)


# ------------------------------------------------------------------------------------------------ Die Ersatzwelt auf einem Gitter


@dataclass(frozen=True)
class World:
    """Kurven auf dem Gitter t: wahr (s_true, Welt A), beobachtbar (s_obs), Ersatzwelt B (s_b: Ereignis, s_c: Ausscheiden)."""

    mech: str
    f: float
    q: float
    t: np.ndarray
    s_true: np.ndarray
    s_obs: np.ndarray
    s_b: np.ndarray
    s_c: np.ndarray

    def at(self, curve: str, x: float) -> float:
        """Wert der Kurve `curve` ("s_true", "s_obs", "s_b", "s_c") bei x (lineare Interpolation auf dem Gitter; vor dem Gitter 1)."""
        return float(np.interp(x, self.t, getattr(self, curve), left=1.0))


def _cumulative_midpoint(fun, x: np.ndarray) -> np.ndarray:
    """Integral von 0 bis x[i] über die Zellen des Gitters (Mittelpunktsregel, Funktionswert in der Zellmitte). Sprungstellen der Funktion liegen auf Gitterpunkten,
    jede Zelle liegt also ganz auf einer Seite des Sprungs."""
    mid = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([[0.0], np.cumsum(fun(mid) * np.diff(x))])


def _grid(mech: str, f: float, q: float) -> np.ndarray:
    """Geometrisches Gitter von GRID_START bis T_GRID_MAX plus 0; beim Mechanismus „Langlieger“ liegen die Sprungstellen d und q·d der Teildichten auf Gitterpunkten."""
    pts = [0.0, *np.geomspace(GRID_START, C.T_GRID_MAX, GRID_POINTS)]
    if mech == C.MECH_LONG and 0.0 < f < 1.0:
        d = S.cut_threshold(f)
        pts += [d, q * d]
    return np.unique(np.array(pts))


def ersatz_world(mech: str, f: float, q: float) -> World:
    """Baut die Ersatzwelt: Hazards aus den beobachtbaren Teildichten, Netto-Überlebensfunktionen als exp(−∫ h) (Mittelpunktsregel auf einem geometrischen Gitter bis T_GRID_MAX)."""
    _check(mech, f, q)
    t = _grid(mech, f, q)

    def hazard(dens):
        def h(x):
            so = np.clip(s_obs(mech, f, q, x), 0.0, 1.0)
            return np.where(so > 1e-300, dens(mech, f, q, x) / np.where(so > 1e-300, so, 1.0), 0.0)
        return h

    so_grid = np.clip(s_obs(mech, f, q, t), 0.0, 1.0)
    return World(mech, f, q, t, S.true_survival(t), so_grid, np.exp(-_cumulative_midpoint(hazard(q_event), t)), np.exp(-_cumulative_midpoint(hazard(q_cens), t)))


def _invert(grid_t: np.ndarray, surv: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Zeit t mit S(t) = v für fallendes S (lineare Interpolation); unendlich, wo v unter dem Grenzwert der Kurve liegt."""
    out = np.interp(v, surv[::-1], grid_t[::-1])
    return np.where(v < surv[-1], np.inf, out)


def sample_world_b(world: World, n: int, seed: int) -> S.Sample:
    """Stichprobe aus der Ersatzwelt B: Ereigniszeit und Ausscheidezeit werden UNABHÄNGIG aus s_b und s_c gezogen (Inversion auf dem Gitter).
    `true_time` ist die Ereigniszeit von B (unendlich, wenn s_b nicht gegen 0 fällt). Beobachtungen hinter dem Gitter werden am Gitterrand zensiert (Wahrscheinlichkeit unter 1e-8)."""
    if n < 1:
        raise ValueError("n muss mindestens 1 sein.")
    rng = SplitMix64(seed)
    v_e = 1.0 - rng.uniform(n)       # in (0, 1]
    v_c = 1.0 - rng.uniform(n)
    t_b = _invert(world.t, world.s_b, v_e)
    t_c = _invert(world.t, world.s_c, v_c)
    obs = np.minimum(np.minimum(t_b, t_c), C.T_GRID_MAX)
    event = (t_b <= t_c) & (t_b <= C.T_GRID_MAX)
    return S.Sample(np.maximum(obs, 1e-9), event, t_b)


def empirical_cifs(sample: S.Sample, grid) -> tuple[np.ndarray, np.ndarray]:
    """Beobachtbare Anteile: Abholung gesehen bis t und ausgeschieden bis t, je Anteil aller Container, an den Stellen grid."""
    g = np.asarray(grid, dtype=float)
    ev = np.sort(sample.obs[sample.event])
    ce = np.sort(sample.obs[~sample.event])
    return np.searchsorted(ev, g, side="right") / sample.n, np.searchsorted(ce, g, side="right") / sample.n


# ------------------------------------------------------------------------------------------------ Tagesraster gegen stetig


def raster_survival(mech: str, f: float, q: float, width: float, times, order: str = "event_first") -> np.ndarray:
    """Kaplan–Meier-Kurve des gerasterten Beobachtungsprozesses (Population, kein Zufall) an den Zeiten `times` (Vielfache der Rasterbreite).
    Beobachtete Zeiten werden auf das Ende ihres Rasterfensters aufgerundet. `order`: „event_first“ (bei Gleichstand im Fenster zählt das Ereignis zuerst,
    die im selben Fenster Ausgeschiedenen bleiben im Risiko: die Konvention von Kaplan–Meier) oder „censor_first“ (sie scheiden vorher aus dem Risiko)."""
    _check(mech, f, q)
    if width <= 0 or order not in ("event_first", "censor_first"):
        raise ValueError("Die Rasterbreite muss positiv und order „event_first“ oder „censor_first“ sein.")
    times = np.asarray(times, dtype=float)
    k_max = int(math.ceil(float(times.max()) / width - 1e-9))
    edges = np.arange(0, k_max + 1, dtype=float) * width
    e = np.diff(cif_event(mech, f, q, edges))              # Ereignisse je Fenster
    c = np.diff(cif_cens(mech, f, q, edges))                # Ausgeschiedene je Fenster
    risk = s_obs(mech, f, q, edges[:-1])                    # im Risiko zu Beginn des Fensters
    den = risk if order == "event_first" else risk - c
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(den > 0, e / den, 0.0)
    surv = np.cumprod(1.0 - h)
    idx = np.rint(times / width).astype(int) - 1
    return surv[idx]
