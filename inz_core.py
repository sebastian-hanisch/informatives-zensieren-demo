"""Kern der Demo: Kaplan–Meier (Kurve, Quantil) mit Eingabeprüfung und der Zufallsstrom SplitMix64.

Kopie aus ueberlebenszeit-planung/kern/uz_core.py, Stand 2026-11-10, gekürzt auf das, was diese Demo braucht (ohne Greenwood-Band, mittlere Lebenszeit,
Log-Rank, konkurrierende Risiken, Weibull, Cox, Schoenfeld). Die kopierten Funktionen sind unverändert (`SplitMix64` ohne `exponential` und `weibull`).

Konventionen: Risikomenge bei t = {i : Eintritt_i < t <= Zeit_i}; bei gleicher Zeit zählt das Ereignis vor der Zensierung (Zensierte bleiben im
Risiko); Zensierung gilt als unabhängig vom Ereignis (diese Demo zeigt, was passiert, wenn sie es nicht ist). Zufall nur über SplitMix64, nie über
`hash()` und nie über numpy-Verteilungen."""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------------------------------------ Zufall

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig.
    Alle Verteilungen sind hier selbst aus Gleichverteilten gebaut (nie numpy-Verteilungen, deren Algorithmen sich zwischen Versionen ändern)."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormal (Box-Muller, nur der Kosinus-Zweig); verbraucht 2·size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)

    def lognormal(self, size: int, median: float, sigma: float) -> np.ndarray:
        """Lognormal mit gegebenem Median und σ der Log-Skala."""
        return median * np.exp(sigma * self.normal(size))


# ------------------------------------------------------------------------------------------------ Eingabeprüfung


class SurvivalError(ValueError):
    """Ungültige Eingabe (verständliche Meldung für die App)."""


def check_data(time, event, entry=None):
    """Prüft und wandelt in float-/bool-Arrays. Zeiten > 0 und endlich, Ereignis 0/1, Eintritt >= 0 und < Zeit (für jede Zeile)."""
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise SurvivalError("Es braucht mindestens eine Beobachtung (eindimensionale Liste der Zeiten).")
    e_raw = np.asarray(event)
    if e_raw.shape != t.shape:
        raise SurvivalError("Zeiten und Ereignisse müssen gleich viele Einträge haben.")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise SurvivalError("Alle Zeiten müssen endlich und größer als 0 sein.")
    if not np.all(np.isin(e_raw, (0, 1, False, True))):
        raise SurvivalError("Das Ereignis muss 0 (zensiert) oder 1 (Ereignis) sein.")
    e = e_raw.astype(bool)
    if entry is None:
        en = np.zeros_like(t)
    else:
        en = np.asarray(entry, dtype=float)
        if en.shape != t.shape or not np.all(np.isfinite(en)) or np.any(en < 0):
            raise SurvivalError("Der Eintritt muss zu jeder Beobachtung angegeben werden, endlich und nicht negativ sein.")
        bad = np.where(en >= t)[0]
        if bad.size:
            raise SurvivalError(f"Der Eintritt muss vor der Zeit liegen (Beobachtung {int(bad[0]) + 1}: Eintritt {en[bad[0]]:g} ≥ Zeit {t[bad[0]]:g}).")
    return t, e, en


# ------------------------------------------------------------------------------------------------ Kaplan–Meier


def kaplan_meier(time, event, entry=None) -> dict:
    """Kaplan–Meier-Schätzer mit verzögertem Eintritt. Gibt Arrays je Ereigniszeit zurück:
    t, n (im Risiko), d (Ereignisse), s (Überleben nach t), gw (Greenwood-Summe Σ d/(n(n−d)), Terme mit n = d fehlen: dort ist s = 0).
    """
    t, e, en = check_data(time, event, entry)
    ts = np.unique(t[e])
    ts_sorted = np.sort(t)
    en_sorted = np.sort(en)
    n = np.searchsorted(en_sorted, ts, side="left") - np.searchsorted(ts_sorted, ts, side="left")  # #(Eintritt < t) − #(Zeit < t)
    d = np.array([np.sum((t == x) & e) for x in ts], dtype=float) if ts.size else np.zeros(0)
    n = n.astype(float)
    if np.any(n < d):
        raise SurvivalError("Interner Fehler: mehr Ereignisse als Beobachtungen im Risiko.")
    s = np.cumprod(1.0 - d / n) if ts.size else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > d, d / (n * (n - d)), 0.0)
    return {"t": ts, "n": n, "d": d, "s": s, "gw": np.cumsum(terms), "n_obs": int(t.size)}


def _index_at(km: dict, x: float) -> int:
    return int(np.searchsorted(km["t"], x, side="right")) - 1


def survival_at(km: dict, x: float) -> float:
    """Ŝ(x); vor der ersten Ereigniszeit 1."""
    i = _index_at(km, x)
    return 1.0 if i < 0 else float(km["s"][i])


def quantile(km: dict, q: float) -> float:
    """Kleinste Ereigniszeit mit Ŝ ≤ 1 − q; nan, wenn die Kurve das Niveau nicht erreicht („nicht schätzbar“)."""
    hit = np.where(km["s"] <= 1.0 - q + 1e-12)[0]
    return float(km["t"][hit[0]]) if hit.size else float("nan")
