"""Kennzahlen, Verdikt-Logik und Auswertung über feste Zufalls-Seeds.

Alle Verdikte sind reine Funktionen (Art, Text) von Zahlen; die App ruft `getattr(st, Art)(Text)`, tests/test_verdicts.py prüft Schwellen und jede Zahl im Text."""

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

import inz_algorithm as A
import inz_constants as C
import inz_core as K
import inz_ersatz as E
import inz_scenario as S

NAN = float("nan")


# ------------------------------------------------------------------------------------------------ Formate


def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma; nan als „nicht schätzbar“, unendlich als „∞“."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nicht schätzbar"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{nd}f}".replace(".", ",")


def days(x: float) -> str:
    """Zeitangabe in Tagen; ein nicht schätzbarer Wert steht ohne „Tage“ („nicht schätzbar“), ein unendlicher als „unendlich“."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nicht schätzbar"
    if isinstance(x, float) and math.isinf(x):
        return "unendlich"
    return f"{de(x)} Tage"


def signed(x: float, nd: int = 2) -> str:
    """Zahl mit Vorzeichen, Dezimalkomma und echtem Minuszeichen."""
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def pct(x: float, nd: int = 1) -> str:
    """Prozentzahl mit Vorzeichen, Dezimalkomma und echtem Minuszeichen."""
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−") + " %"


def deviation_percent(value: float, truth: float) -> float:
    """Abweichung in Prozent der WAHRHEIT (Basis ist die Wahrheit, nicht die Schätzung); nan bleibt nan."""
    if value is None or math.isnan(value) or truth == 0:
        return NAN
    return 100.0 * (value - truth) / truth


def fraction_text(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# ------------------------------------------------------------------------------------------------ Einstellungen und Analyse


@dataclass(frozen=True)
class Settings:
    mech: str = C.MECH_NEAR
    f_pct: int = C.DEFAULT_F
    q_pct: int = C.DEFAULT_Q
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED

    @property
    def f(self) -> float:
        return self.f_pct / 100.0

    @property
    def q(self) -> float:
        return self.q_pct / 100.0


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    sample: S.Sample
    bounds: A.Bounds
    true_median: float          # Median der Population (4 Tage)
    sample_truth: float         # Median der wahren Zeiten dieser Stichprobe


def analyse(settings: Settings) -> Analysis:
    sample = S.simulate(settings.mech, settings.f, settings.q, settings.n, settings.seed)
    return Analysis(settings, sample, A.bounds(sample.obs, sample.event), C.MEDIAN_TRUE, A.sample_median_true(sample.true_time))


def repeat_summary(settings: Settings, seeds=C.REPEAT_SEEDS) -> dict:
    """Mediane von Kaplan–Meier, unterer und oberer Schranke und die Wahrheit der Stichprobe über die festen Zufalls-Seeds (nan bleibt nan)."""
    rows = {"km": [], "lower": [], "upper": [], "sample_truth": [], "censored": []}
    for seed in seeds:
        s = S.simulate(settings.mech, settings.f, settings.q, settings.n, seed)
        b = A.bounds(s.obs, s.event)
        rows["km"].append(b.km.median)
        rows["lower"].append(b.lower.median)
        rows["upper"].append(b.upper.median)
        rows["sample_truth"].append(A.sample_median_true(s.true_time))
        rows["censored"].append(s.censored_share)
    return {"seeds": len(seeds), **rows}


def spread(values) -> dict:
    """Median, Minimum und Maximum der definierten Werte, dazu wie viele definiert sind."""
    a = np.asarray(values, dtype=float)
    ok = a[~np.isnan(a)]
    if ok.size == 0:
        return {"median": NAN, "low": NAN, "high": NAN, "defined": 0, "of": int(a.size)}
    return {"median": float(np.median(ok)), "low": float(ok.min()), "high": float(ok.max()), "defined": int(ok.size), "of": int(a.size)}


@lru_cache(maxsize=1)
def measurements() -> dict:
    """Eingefrorene Messung (tools/measure_informatives_zensieren.py); nicht lesbar -> leeres Ergebnis, die App zeigt dann einen Hinweis statt einer Tabelle."""
    path = Path(__file__).resolve().parent / C.MEASURE_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------------------------------------ Handbeispiel mit Brüchen

HAND_ASSUMPTIONS = ("Kaplan–Meier", "Alle Zensierten gehen sofort ab", "Alle Zensierten bleiben für immer")
_FOREVER = 10**9


def hand_rows(assumption: str, times=S.HAND_TIMES, events=S.HAND_EVENTS) -> list[dict]:
    """Kaplan–Meier mit exakten Brüchen für das Handbeispiel unter einer Annahme über die Zensierten: je Ereigniszeit t, im Risiko n, Ereignisse d, Faktor (n − d)/n und S(t).
    Konvention: bei gleicher Zeit zählt das Ereignis vor der Zensierung; „gehen sofort ab“ macht jeden Zensierten zum Ereignis bei seiner Zeit, „bleiben für immer“ setzt ihn ans Ende."""
    if assumption not in HAND_ASSUMPTIONS:
        raise ValueError(f"Unbekannte Annahme: {assumption!r}")
    if assumption == HAND_ASSUMPTIONS[1]:
        times, events = list(times), [1] * len(times)
    elif assumption == HAND_ASSUMPTIONS[2]:
        times, events = [x if e else _FOREVER for x, e in zip(times, events)], list(events)
    rows, s = [], Fraction(1)
    for t in sorted({x for x, e in zip(times, events) if e}):
        n = sum(1 for x in times if x >= t)
        d = sum(1 for x, e in zip(times, events) if x == t and e)
        factor = Fraction(n - d, n)
        s *= factor
        rows.append({"t": t, "n": n, "d": d, "factor": factor, "s": s})
    return rows


def hand_median(rows: list[dict]):
    """Kleinste Ereigniszeit mit S <= 1/2; None, wenn es keine gibt."""
    return next((r["t"] for r in rows if r["s"] <= Fraction(1, 2)), None)


# ------------------------------------------------------------------------------------------------ Experiment 3: zwei Welten




def curve_median(world: E.World, curve: str, level: float = 0.5) -> float:
    """Erste Gitterzeit, bei der die Kurve höchstens `level` erreicht (lineare Interpolation); nan, wenn sie es nie tut."""
    y = getattr(world, curve)
    hit = np.where(y <= level)[0]
    if hit.size == 0:
        return NAN
    i = int(hit[0])
    if i == 0:
        return float(world.t[0])
    y0, y1, t0, t1 = y[i - 1], y[i], world.t[i - 1], world.t[i]
    return float(t0 + (y0 - level) / (y0 - y1) * (t1 - t0)) if y0 != y1 else float(t1)


def world_summary(settings: Settings, n_world: int, seed: int) -> dict:
    """Ersatzwelt, je eine Stichprobe aus Welt A (abhängig) und Welt B (unabhängig) und der Vergleich der beobachtbaren Anteile.
    `diff_aa` ist derselbe Vergleich zwischen zwei Stichproben aus Welt A (gleiches n, anderer Zufalls-Seed) als Maß für den Zufall."""
    world = E.ersatz_world(settings.mech, settings.f, settings.q)
    sa = S.simulate(settings.mech, settings.f, settings.q, n_world, seed)
    sb = E.sample_world_b(world, n_world, seed + 1)
    sa2 = S.simulate(settings.mech, settings.f, settings.q, n_world, seed + 2)
    grid = np.linspace(0.0, 30.0, 301)
    ea, ca = E.empirical_cifs(sa, grid)
    eb, cb = E.empirical_cifs(sb, grid)
    ea2, ca2 = E.empirical_cifs(sa2, grid)
    return {
        "world": world, "sample_a": sa, "sample_b": sb, "grid": grid, "cif_a": (ea, ca), "cif_b": (eb, cb),
        "diff_ab": float(max(np.abs(ea - eb).max(), np.abs(ca - cb).max())),
        "diff_aa": float(max(np.abs(ea - ea2).max(), np.abs(ca - ca2).max())),
        "km_a": A.make_curve(K.kaplan_meier(sa.obs, sa.event)),
        "km_b": A.make_curve(K.kaplan_meier(sb.obs, sb.event)),
        "median_a": C.MEDIAN_TRUE, "median_b": curve_median(world, "s_b"),
    }


def world_rows(w: dict, times=C.WORLD_TIMES) -> list[dict]:
    """Je Zeitpunkt: wahre Welt A, Ersatzwelt B (geschlossene Form), Kaplan–Meier aus den Daten von A und aus den Daten von B."""
    return [{"t": t, "s_true": w["world"].at("s_true", t), "s_b": w["world"].at("s_b", t),
             "km_a": K.survival_at(w["km_a"].km, t), "km_b": K.survival_at(w["km_b"].km, t)} for t in times]


# ------------------------------------------------------------------------------------------------ Verdikte (reine Funktionen)


def verdict_hand(assumption: str, rows: list[dict], median) -> tuple[str, str]:
    """Verdikt des Handbeispiels zu einer Annahme über die zwei Zensierten (3+ und 5+)."""
    steps = "; ".join(f"S({r['t']}) = {fraction_text(r['s'])}" for r in rows)
    med = f"Median {median} Tage" if median is not None else "die Kurve erreicht 1/2 nicht: kein Median"
    return "info", f"{assumption}: {steps}. {med}."


def verdict_core(a: Analysis) -> tuple[str, str]:
    """Kernfrage: Wie weit kann Kaplan–Meier danebenliegen, und was sagen die Schranken? Nennt nur Zahlen dieser einen Stichprobe (Basis der Prozente ist die Wahrheit)."""
    s, b = a.sample, a.bounds
    if s.n_events == 0:
        return "warning", (f"Es wurde keine einzige Abholung beobachtet: alle {s.n} Container sind vorzeitig ausgeschieden. Kaplan–Meier hat nichts zu schätzen, "
                           f"die obere Schranke (alle bleiben für immer) ist unendlich, und die untere (alle gehen sofort ab) liegt bei {de(b.lower.median)} Tagen.")
    if s.n_events == s.n:
        return "info", (f"In dieser Stichprobe ist niemand vorzeitig ausgeschieden: Kaplan–Meier und beide Schranken sind dieselbe Kurve (Median {de(b.km.median)} Tage; "
                        f"wahrer Median {de(a.true_median)}). Es gibt nichts, was die Schranken öffnen könnte.")
    share = de(100 * s.censored_share, 1)
    head = f"{share} % der Container sind vorzeitig ausgeschieden. "
    truth = (f"Der Median der wahren Zeiten dieser Stichprobe ist {de(a.sample_truth)} Tage; er liegt "
             + ("zwischen den Schranken." if A.inside(b.lower.median, b.upper.median, a.sample_truth) else "außerhalb der Schranken (das darf nicht vorkommen)."))
    up = f"{de(b.upper.median)} Tage" if not math.isnan(b.upper.median) else "unendlich (auch wenn alle Ausgeschiedenen für immer bleiben, fiele die Kurve nicht auf 0,5)"
    if b.km.status != A.STATUS_OK:
        return "warning", head + f"Die Kaplan–Meier-Kurve erreicht 0,5 nicht: Der Median ist nicht schätzbar. Untere Schranke {de(b.lower.median)} Tage, obere Schranke {up}. " + truth
    dev = deviation_percent(b.km.median, a.true_median)
    side = "über" if b.km.median > a.true_median else "unter"
    text = (head + f"Kaplan–Meier schätzt den Median auf {de(b.km.median)} Tage (wahr {de(a.true_median)}: {pct(dev)}, also {side} der Wahrheit). "
            f"Untere Schranke {de(b.lower.median)} Tage, obere Schranke {up}. " + truth)
    return ("warning" if abs(dev) >= 10.0 else "info"), text


def verdict_repeat(rep: dict, true_median: float, mech: str, n: int) -> tuple[str, str]:
    """Experiment „Anteil f“ über die festen Zufalls-Seeds: Median der Schätzungen mit Spanne, nicht schätzbare Werte genannt, Lage der Wahrheit der Stichprobe."""
    km, lo, up = spread(rep["km"]), spread(rep["lower"]), spread(rep["upper"])
    runs = rep["seeds"]
    above = sum(1 for v in rep["km"] if not math.isnan(v) and v > true_median)
    below = sum(1 for v in rep["km"] if not math.isnan(v) and v < true_median)
    enclosed = sum(1 for a_, b_, c_ in zip(rep["lower"], rep["upper"], rep["sample_truth"]) if A.inside(a_, b_, c_))
    text = (f"{runs} feste Zufalls-Seeds, Mechanismus „{mech}“, n = {n}: Median der Läufe bei Kaplan–Meier {days(km['median'])} (Spanne {de(km['low'])} bis {de(km['high'])}; wahrer Median {de(true_median)}), "
            f"untere Schranke {de(lo['median'])}, obere Schranke {de(up['median'])}. Kaplan–Meier liegt in {above} von {runs} Läufen über dem wahren Median, in {below} darunter. "
            f"Die wahren Zeiten der jeweiligen Stichprobe liegen in {enclosed} von {runs} Läufen zwischen den Schranken.")
    if up["defined"] < up["of"]:
        text += f" Die obere Schranke ist in {up['of'] - up['defined']} von {runs} Läufen nicht endlich; der Wert oben gilt nur für die übrigen {up['defined']}."
    if km["defined"] < km["of"]:
        text += f" Kaplan–Meier hat in {km['of'] - km['defined']} von {runs} Läufen keinen Median."
    return "info", text


def verdict_phi(tip: dict, threshold: float, km_median: float, lower_median: float) -> tuple[str, str]:
    """Experiment „Kipppunkt“: Wie viele der Ausgeschiedenen müssten in Wahrheit sofort abgehen, damit der Schluss „Median ≥ Schwelle“ nicht mehr gilt?"""
    thr = de(threshold, 1)
    if tip["status"] == A.TIP_ALREADY:
        return "warning", f"Der Schluss „Median ≥ {thr} Tage“ gilt in dieser Stichprobe schon bei φ = 0 nicht: Der Kaplan–Meier-Median ist {de(km_median)} Tage."
    if tip["status"] == A.TIP_NEVER:
        return "success", (f"Der Schluss „Median ≥ {thr} Tage“ bleibt bis φ = 1 bestehen: Selbst wenn alle Ausgeschiedenen sofort abgegangen wären (untere Schranke), läge der Median bei {de(lower_median)} Tagen. "
                           "In dieser Stichprobe hängt der Schluss nicht an der Annahme über die Ausgeschiedenen.")
    phi = tip["phi_star"]
    return "warning", (f"Der Schluss „Median ≥ {thr} Tage“ gilt für φ = 0 (Kaplan–Meier: {days(km_median)}) und kippt bei φ* = {de(phi)}: "
                       f"Wären schon {de(100 * phi, 0)} % der Ausgeschiedenen in Wahrheit sofort abgegangen, läge der Median unter {thr} Tagen (Raster 0,01).")


def verdict_world(rows: list[dict], median_a: float, median_b: float, diff_ab: float, diff_aa: float, same_world: bool) -> tuple[str, str]:
    """Experiment „Zwei Welten, dieselben Daten“: wahre Kurve gegen Ersatzwelt gegen Kaplan–Meier aus beiden Datensätzen."""
    if same_world:
        return "info", ("Es scheidet niemand vorzeitig aus (oder die Ausgeschiedenen ändern die beobachtbaren Daten nicht): Welt B ist Welt A, "
                        f"die Kurven fallen zusammen, und der Median ist in beiden {de(median_a)} Tage.")
    parts = "; ".join(f"t = {r['t']}: wahr {de(r['s_true'], 3)}, Ersatzwelt {de(r['s_b'], 3)}, Kaplan–Meier aus A {de(r['km_a'], 3)}, aus B {de(r['km_b'], 3)}" for r in rows)
    return "info", (f"Überlebensanteile: {parts}. Wahrer Median Welt A {days(median_a)}, Welt B {days(median_b) if not math.isnan(median_b) else 'unendlich (die Kurve fällt nie auf 0,5)'}. "
                    f"Größter Unterschied der beobachtbaren Anteile zwischen den Stichproben aus A und B: {de(diff_ab, 3)}; zwischen zwei Stichproben aus A: {de(diff_aa, 3)}.")


# ------------------------------------------------------------------------------------------------ Tabellen aus der eingefrorenen Messung


def _cell(d: dict, runs: int) -> str:
    """Mittel ± Standardfehler der definierten Läufe; fehlen Läufe (kein endlicher Wert), steht dabei, in wie vielen er endlich ist."""
    if d["mean"] is None:
        return f"in keinem von {runs} Läufen endlich"
    text = f"{de(d['mean'])} ± {de(d['se'], 3)}"
    return text if d["undefined"] == 0 else f"{text} (endlich in {runs - d['undefined']} von {runs} Läufen)"


def bounds_table(meas: dict) -> list[dict]:
    """Zeilen der Tabelle „Kaplan–Meier und Schranken über viele Stichproben“ (Experiment 2) aus der Messung."""
    rows = []
    for c in meas.get("bounds_cells", []):
        runs = c["runs"]
        rows.append({
            "Mechanismus": c["mech"], "f": f"{round(100 * c['f'])} %", "q": f"{round(100 * c['q'])} %", "ausgeschieden": f"{de(100 * c['censored']['mean'], 1)} %",
            "Median Kaplan–Meier": _cell(c["km"], runs), "untere Schranke": _cell(c["lower"], runs), "obere Schranke": _cell(c["upper"], runs),
            "Kaplan–Meier über 4,00": f"{c['km_above_truth']} von {c['km_defined']}",
            "wahre Zeiten der Stichprobe in den Schranken": f"{c['inside_sample']} von {runs}", "Median 4,00 in den Schranken": f"{c['inside_population']} von {runs}",
        })
    return rows


def tip_table(meas: dict) -> list[dict]:
    """Zeilen der Tabelle „Kipppunkt“ (Experiment 4) aus der Messung."""
    rows = []
    for c in meas.get("tip_cells", []):
        st, runs = c["status"], c["runs"]
        med = c["median_at_phi"]
        rows.append({
            "f": f"{round(100 * c['f'])} %", "Median bei φ = 0 / 0,25 / 0,5 / 0,75 / 1": " / ".join(de(med[k]["mean"]) for k in ("0.0", "0.25", "0.5", "0.75", "1.0")),
            "Schluss gilt schon bei φ = 0 nicht": f"{st[A.TIP_ALREADY]} von {runs}", "kippt bei φ*": f"{st[A.TIP_TIPS]} von {runs}, im Mittel φ* = {de(c['phi_star']['mean'])} ± {de(c['phi_star']['se'], 3)}",
            "kippt nie": f"{st[A.TIP_NEVER]} von {runs}",
        })
    return rows


def raster_rows(mech: str, f: float, q: float, times=C.WORLD_TIMES, widths=C.RASTERS) -> list[dict]:
    """Tagesraster gegen stetig (Rechnung ohne Zufall): je Rasterbreite die gerasterte Kurve an den Zeiten, mit beiden Gleichstandsregeln."""
    return [{"width": w, "event_first": [float(x) for x in E.raster_survival(mech, f, q, w, times, "event_first")],
             "censor_first": [float(x) for x in E.raster_survival(mech, f, q, w, times, "censor_first")]} for w in widths]


def raster_table(rasters: list[dict], times=C.WORLD_TIMES) -> list[dict]:
    """Zeilen der Tabelle „Tagesraster gegen stetig“: Überleben der gerasterten Kurve an den Zeiten, je Gleichstandsregel."""
    rows = []
    for r in rasters:
        rows.append({"Rasterbreite (Tage)": de(r["width"], 2), **{f"Ereignis zuerst, t = {t}": de(v, 3) for t, v in zip(times, r["event_first"])},
                     **{f"Zensierung zuerst, t = {t}": de(v, 3) for t, v in zip(times, r["censor_first"])}})
    return rows
