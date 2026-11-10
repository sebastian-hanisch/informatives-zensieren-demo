"""Szenario: Containerverweilzeit D lognormal (Median 4 Tage, σ = 0,9) und zwei Mechanismen, nach denen ein Teil der Container vorzeitig aus der Beobachtung ausscheidet.

Die Erzeuger sind reine Funktionen von (Mechanismus, f, q, n, Zufalls-Seed); der Zufall kommt nur aus SplitMix64 (inz_core).
Ausscheiden heißt: Die Beobachtung des Containers endet bei q·D, bevor er abgeholt worden wäre (q < 1); er zählt dann als zensiert.
Die Wahrheit (Überlebensfunktion, Quantile) steht in geschlossener Form hier; die Formeln des beobachtbaren Teils (Ersatzwelt) stehen in inz_ersatz.py."""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

import inz_constants as C
from inz_core import SplitMix64

_MU = math.log(C.MEDIAN_TRUE)
_NORMAL = NormalDist()


@dataclass(frozen=True)
class Sample:
    """Beobachtete Zeiten (Tage) und Ereignis (True = Abholung gesehen, False = vorzeitig ausgeschieden); `true_time` ist die wahre Zeit bis zur Abholung (in echten Daten unbekannt)."""

    obs: np.ndarray
    event: np.ndarray
    true_time: np.ndarray

    @property
    def n(self) -> int:
        return int(self.obs.size)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    @property
    def censored_share(self) -> float:
        return 1.0 - self.n_events / self.n


def cut_threshold(f: float) -> float:
    """Verweilzeit d mit P(D > d) = f (Population): Beim Mechanismus „Langlieger“ scheiden die Container mit D > d aus. f = 0 ergibt unendlich, f = 1 ergibt 0."""
    if f <= 0.0:
        return math.inf
    if f >= 1.0:
        return 0.0
    return math.exp(_MU + C.SIGMA_TRUE * _NORMAL.inv_cdf(1.0 - f))


def simulate(mech: str, f: float, q: float, n: int, seed: int) -> Sample:
    """Eine Stichprobe von n Containern.
    Kurz vor der Abholung: Jeder Container scheidet unabhängig mit Wahrscheinlichkeit f aus. Langlieger: Es scheidet aus, wer länger bleibt als die Schwelle `cut_threshold(f)`.
    Wer ausscheidet, wird bei q·D zensiert; sonst wird die Abholung bei D gesehen. Die Verweilzeiten D sind bei gleichem Zufalls-Seed in beiden Mechanismen dieselben."""
    if mech not in C.MECHANISMS:
        raise ValueError(f"Unbekannter Mechanismus: {mech!r}")
    if not (0.0 <= f <= 1.0) or not (0.0 < q < 1.0) or n < 1:
        raise ValueError("f muss in [0, 1], q in (0, 1) und n mindestens 1 sein.")
    rng = SplitMix64(seed)
    d = rng.lognormal(n, C.MEDIAN_TRUE, C.SIGMA_TRUE)
    u = rng.uniform(n)
    cut = (u < f) if mech == C.MECH_NEAR else (d > cut_threshold(f))
    return Sample(np.where(cut, q * d, d), ~cut, d)


# ------------------------------------------------------------------------------------------------ Wahrheit (geschlossene Formeln)


def true_survival(t):
    """S(t) = 1 − Φ((ln t − ln Median)/σ); S(t) = 1 für t <= 0."""
    t = np.asarray(t, dtype=float)
    z = np.where(t > 0, (np.log(np.where(t > 0, t, 1.0)) - _MU) / C.SIGMA_TRUE, -np.inf)
    return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))


def true_density(t):
    """Dichte g(t) der lognormalen Verweilzeit; 0 für t <= 0."""
    t = np.asarray(t, dtype=float)
    tp = np.where(t > 0, t, 1.0)
    z = (np.log(tp) - _MU) / C.SIGMA_TRUE
    return np.where(t > 0, np.exp(-0.5 * z * z) / (tp * C.SIGMA_TRUE * math.sqrt(2.0 * math.pi)), 0.0)


def true_quantile(q: float) -> float:
    """Wahres q-Quantil der Verweilzeit (q = 0,5 ergibt den Median 4)."""
    return math.exp(_MU + C.SIGMA_TRUE * _NORMAL.inv_cdf(q))


def true_mean() -> float:
    return math.exp(_MU + C.SIGMA_TRUE**2 / 2)


# ------------------------------------------------------------------------------------------------ Handbeispiel (zwei Zensierte unter sechs Containern)


HAND_TIMES = (2, 3, 4, 5, 5, 8)
HAND_EVENTS = (1, 0, 1, 1, 0, 1)    # 3+ und 5+ sind vorzeitig ausgeschieden
