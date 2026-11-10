"""Verfahren der Demo: Kaplan–Meier mit zwei Schranken, gewichtete Kurve für die Sensitivitätsannahme φ und der Kipppunkt.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); der Kern (Kaplan–Meier, Quantil) steht in inz_core.py.

Die zwei Schranken behandeln die Zensierten extrem:
  untere Schranke: jeder Zensierte geht in Wahrheit sofort ab (sein Ereignis liegt bei seiner Zensierzeit),
  obere Schranke:  jeder Zensierte bleibt für immer (sein Ereignis tritt nie ein, er bleibt bis zum Schluss im Risiko).
Beide gelten für jede Zensierung, die vor dem Ereignis liegt, auch für eine abhängige; sie sagen nichts darüber, wo zwischen ihnen die Wahrheit liegt.
Die Sensitivitätsannahme φ legt fest, welcher Anteil der Zensierten in Wahrheit sofort abgeht: φ = 0 ist Kaplan–Meier, φ = 1 die untere Schranke."""

import math
from dataclasses import dataclass

import numpy as np

import inz_constants as C
import inz_core as K

STATUS_OK = "ok"
STATUS_NOT_ESTIMABLE = "nicht_schaetzbar"       # die Kurve erreicht 0,5 nicht
STATUS_NO_EVENTS = "keine_ereignisse"           # in den Daten ist keine Abholung zu sehen
NAN = float("nan")


@dataclass(frozen=True)
class Curve:
    """Eine Kaplan–Meier-Kurve (km wie in inz_core.kaplan_meier) mit ihrem Median und dem Status des Medians."""

    km: dict
    median: float
    status: str


def make_curve(km: dict) -> Curve:
    """Median der Kurve: kleinste Ereigniszeit mit Ŝ ≤ 0,5; nan mit Status, wenn es keinen gibt."""
    if km["t"].size == 0:
        return Curve(km, NAN, STATUS_NO_EVENTS)
    med = K.quantile(km, 0.5)
    return Curve(km, med, STATUS_OK if not math.isnan(med) else STATUS_NOT_ESTIMABLE)


def lower_bound_km(obs, event) -> dict:
    """Untere Schranke: Jeder Zensierte geht bei seiner Zensierzeit ab, alle Zeiten sind Ereignisse (dann ist Kaplan–Meier der Anteil der Zeiten > t)."""
    t, _, _ = K.check_data(obs, event)
    return K.kaplan_meier(t, np.ones(t.size, dtype=bool))


def upper_bound_km(obs, event) -> dict:
    """Obere Schranke: Jeder Zensierte bleibt für immer. Seine Zeit wird hinter die größte beobachtete Zeit gesetzt und er bleibt zensiert, also in jeder Risikomenge."""
    t, e, _ = K.check_data(obs, event)
    late = float(t.max()) + 1.0
    return K.kaplan_meier(np.where(e, t, late), e)


@dataclass(frozen=True)
class Bounds:
    km: Curve
    lower: Curve
    upper: Curve


def bounds(obs, event) -> Bounds:
    """Kaplan–Meier und beide Schranken aus (Zeit, Ereignis)."""
    return Bounds(make_curve(K.kaplan_meier(obs, event)), make_curve(lower_bound_km(obs, event)), make_curve(upper_bound_km(obs, event)))


def sample_median_true(true_time) -> float:
    """Median der wahren Zeiten einer Stichprobe (kleinste Zeit, bei der höchstens die Hälfte länger bleibt): die Wahrheit, die die Schranken sicher einschließen."""
    t = np.asarray(true_time, dtype=float)
    return K.quantile(K.kaplan_meier(t, np.ones(t.size, dtype=bool)), 0.5)


def inside(lower: float, upper: float, truth: float) -> bool:
    """Liegt die Wahrheit zwischen den Medianen der Schranken? Eine nicht schätzbare obere Schranke gilt als unendlich."""
    hi = math.inf if math.isnan(upper) else upper
    return bool(lower - 1e-12 <= truth <= hi + 1e-12)


def survival_values(km: dict, x) -> np.ndarray:
    """Ŝ an den Stellen x (vektorisiert; vor der ersten Ereigniszeit 1)."""
    x = np.asarray(x, dtype=float)
    if km["t"].size == 0:
        return np.ones_like(x)
    idx = np.searchsorted(km["t"], x, side="right") - 1
    return np.where(idx < 0, 1.0, km["s"][np.maximum(idx, 0)])


# ------------------------------------------------------------------------------------------------ Sensitivitätsannahme φ


def weighted_km(time, event, weight) -> dict:
    """Kaplan–Meier mit Gewichten (Anteilen): n_j = Summe der Gewichte mit Zeit >= t_j, d_j = Summe der Gewichte der Ereignisse bei t_j.
    Zeilen mit Gewicht 0 zählen nicht. Gibt t und s (wie `inz_core.kaplan_meier`, ohne n, d, gw)."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    w = np.asarray(weight, dtype=float)
    if t.ndim != 1 or t.size == 0 or e.shape != t.shape or w.shape != t.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise K.SurvivalError("Zeiten, Ereignisse und Gewichte müssen gleich viele Einträge haben; Gewichte nicht negativ.")
    live = e & (w > 0)
    ts, inv = np.unique(t[live], return_inverse=True)
    if ts.size == 0:
        return {"t": ts, "s": np.zeros(0)}
    d = np.bincount(inv, weights=w[live], minlength=ts.size)
    order = np.argsort(t, kind="stable")
    suffix = np.cumsum(w[order][::-1])[::-1]          # Summe der Gewichte ab Position i (Zeit aufsteigend)
    n = suffix[np.searchsorted(t[order], ts, side="left")]
    return {"t": ts, "s": np.cumprod(1.0 - d / n)}


def phi_split(obs, event, phi: float):
    """Zerlegt jeden Zensierten in einen Anteil φ, der bei seiner Zensierzeit abgeht (Ereignis), und einen Anteil 1 − φ, der zensiert bleibt.
    Gibt (Zeit, Ereignis, Gewicht) der verdoppelten Zeilenliste zurück."""
    if not 0.0 <= phi <= 1.0:
        raise K.SurvivalError("φ muss zwischen 0 und 1 liegen.")
    t, e, _ = K.check_data(obs, event)
    c = ~e
    time = np.concatenate([t, t[c]])
    ev = np.concatenate([e, np.ones(int(c.sum()), dtype=bool)])
    w = np.concatenate([np.where(c, 1.0 - phi, 1.0), np.full(int(c.sum()), phi)])
    return time, ev, w


def median_given_phi(obs, event, phi: float) -> float:
    """Median der Kurve, wenn der Anteil φ der Zensierten in Wahrheit sofort abgeht (nan, wenn die Kurve 0,5 nicht erreicht)."""
    t, ev, w = phi_split(obs, event, phi)
    return K.quantile(weighted_km(t, ev, w), 0.5)


TIP_ALREADY = "gilt_schon_nicht"    # der Schluss gilt schon bei φ = 0 nicht
TIP_TIPS = "kippt"                  # der Schluss gilt bei φ = 0 und kippt bei φ*
TIP_NEVER = "kippt_nie"             # der Schluss gilt bei jedem φ bis 1


def tipping_point(obs, event, threshold: float, grid: int = C.PHI_GRID) -> dict:
    """Kleinstes φ auf dem Raster 0, 1/(grid−1), …, 1, bei dem der Schluss „Median ≥ threshold“ nicht mehr gilt.
    Ein nicht schätzbarer Median (Kurve erreicht 0,5 nicht) zählt als „Median ≥ threshold“. Gibt status, phi_star (nan, wenn es keines gibt) und die Mediane auf dem Raster."""
    if grid < 2:
        raise ValueError("Das Raster braucht mindestens zwei Punkte.")
    phis = np.linspace(0.0, 1.0, grid)
    meds = np.array([median_given_phi(obs, event, float(p)) for p in phis])
    holds = np.isnan(meds) | (meds >= threshold - 1e-12)
    if not holds[0]:
        return {"status": TIP_ALREADY, "phi_star": 0.0, "phis": phis, "medians": meds}
    broken = np.where(~holds)[0]
    if broken.size == 0:
        return {"status": TIP_NEVER, "phi_star": NAN, "phis": phis, "medians": meds}
    return {"status": TIP_TIPS, "phi_star": float(phis[broken[0]]), "phis": phis, "medians": meds}
