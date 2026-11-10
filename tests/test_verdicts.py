"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Seeds. Damit prüft der Test die Grenzen (Abweichung ab genau 10 % ist eine Warnung) und jede genannte Zahl direkt."""

import math
from fractions import Fraction as F

import numpy as np

import inz_algorithm as A
import inz_evaluation as ev
import inz_scenario as S

NAN = math.nan


def _curve(median, status=A.STATUS_OK):
    return A.Curve({}, median, status)


def _analysis(km, lo, up, n=10, events=7, true_median=5.0, sample_truth=4.2, km_status=A.STATUS_OK):
    """Hand gebaute Analyse: nur Mediane, Status und Zählungen gehen in `verdict_core` ein."""
    obs = np.arange(1.0, n + 1)
    sample = S.Sample(obs, np.arange(n) < events, obs)
    up_status = A.STATUS_OK if not math.isnan(up) else A.STATUS_NOT_ESTIMABLE
    return ev.Analysis(ev.Settings(), sample, A.Bounds(_curve(km, km_status), _curve(lo), _curve(up, up_status)), true_median, sample_truth)


def test_formate():
    assert ev.de(4.0) == "4,00" and ev.de(NAN) == "nicht schätzbar" and ev.de(math.inf) == "∞"
    assert ev.days(4.0) == "4,00 Tage" and ev.days(NAN) == "nicht schätzbar" and ev.days(math.inf) == "unendlich"
    assert ev.signed(-0.5) == "−0,50" and ev.signed(1.25) == "+1,25"
    assert ev.pct(-12.34) == "−12,3 %" and ev.pct(9.96) == "+10,0 %"
    assert ev.deviation_percent(5.5, 5.0) == 10.0 and math.isnan(ev.deviation_percent(NAN, 4.0)) and math.isnan(ev.deviation_percent(4.0, 0.0))
    assert ev.fraction_text(F(5, 6)) == "5/6" and ev.fraction_text(F(0)) == "0" and ev.fraction_text(F(2, 1)) == "2"


def test_verdict_hand_nennt_faktoren_und_median():
    rows = ev.hand_rows(ev.HAND_ASSUMPTIONS[2])
    kind, text = ev.verdict_hand(ev.HAND_ASSUMPTIONS[2], rows, ev.hand_median(rows))
    assert kind == "info"
    assert text == "Alle Zensierten bleiben für immer: S(2) = 5/6; S(4) = 2/3; S(5) = 1/2; S(8) = 1/3. Median 5 Tage."
    kind, text = ev.verdict_hand("Test", [], None)
    assert "kein Median" in text


def test_hand_zeilen_alle_drei_annahmen():
    km, lo, up = (ev.hand_rows(a) for a in ev.HAND_ASSUMPTIONS)
    assert [(r["t"], r["n"], r["d"], r["s"]) for r in km] == [(2, 6, 1, F(5, 6)), (4, 4, 1, F(5, 8)), (5, 3, 1, F(5, 12)), (8, 1, 1, F(0))]
    assert [(r["t"], r["n"], r["d"], r["s"]) for r in lo] == [(2, 6, 1, F(5, 6)), (3, 5, 1, F(2, 3)), (4, 4, 1, F(1, 2)), (5, 3, 2, F(1, 6)), (8, 1, 1, F(0))]
    assert [(r["t"], r["n"], r["d"], r["s"]) for r in up] == [(2, 6, 1, F(5, 6)), (4, 5, 1, F(2, 3)), (5, 4, 1, F(1, 2)), (8, 3, 1, F(1, 3))]
    assert (ev.hand_median(km), ev.hand_median(lo), ev.hand_median(up)) == (5, 4, 5)
    assert ev.hand_median([]) is None


def test_verdict_core_zahlen_und_richtung():
    kind, text = ev.verdict_core(_analysis(km=6.0, lo=3.5, up=9.0, events=7, sample_truth=4.2))      # +20,0 % über der Wahrheit 5,0
    assert kind == "warning"
    assert "30,0 % der Container sind vorzeitig ausgeschieden" in text
    assert "Kaplan–Meier schätzt den Median auf 6,00 Tage (wahr 5,00: +20,0 %, also über der Wahrheit)" in text
    assert "Untere Schranke 3,50 Tage, obere Schranke 9,00 Tage" in text
    assert "Der Median der wahren Zeiten dieser Stichprobe ist 4,20 Tage; er liegt zwischen den Schranken." in text
    kind, text = ev.verdict_core(_analysis(km=3.0, lo=2.0, up=NAN))                                    # −40 %: unter der Wahrheit, obere Schranke unendlich
    assert kind == "warning" and "(wahr 5,00: −40,0 %, also unter der Wahrheit)" in text and "obere Schranke unendlich" in text


def test_verdict_core_schwelle_zehn_prozent_ist_inklusive():
    assert ev.verdict_core(_analysis(km=5.5, lo=4.0, up=7.0))[0] == "warning"        # genau +10,0 %
    assert ev.verdict_core(_analysis(km=5.45, lo=4.0, up=7.0))[0] == "info"          # +9,0 %
    assert ev.verdict_core(_analysis(km=4.5, lo=4.0, up=7.0))[0] == "warning"        # genau −10,0 %
    assert ev.verdict_core(_analysis(km=4.55, lo=4.0, up=7.0))[0] == "info"


def test_verdict_core_wahrheit_ausserhalb_der_schranken_wird_gemeldet():
    _, text = ev.verdict_core(_analysis(km=6.0, lo=3.5, up=9.0, sample_truth=9.5))
    assert "außerhalb der Schranken" in text


def test_verdict_core_kaplan_meier_ohne_median_und_ohne_abholung_und_ohne_ausscheiden():
    kind, text = ev.verdict_core(_analysis(km=NAN, lo=3.0, up=NAN, km_status=A.STATUS_NOT_ESTIMABLE))
    assert kind == "warning" and "erreicht 0,5 nicht" in text and "Untere Schranke 3,00 Tage" in text
    kind, text = ev.verdict_core(_analysis(km=NAN, lo=3.0, up=NAN, events=0, km_status=A.STATUS_NO_EVENTS))
    assert kind == "warning" and "keine einzige Abholung" in text and "alle 10 Container" in text and "liegt bei 3,00 Tagen" in text
    kind, text = ev.verdict_core(_analysis(km=5.0, lo=5.0, up=5.0, events=10))
    assert kind == "info" and "niemand vorzeitig ausgeschieden" in text and "Median 5,00 Tage" in text and "wahrer Median 5,00" in text


def test_verdict_repeat_zaehlungen_und_hinweise():
    rep = {"seeds": 4, "km": [5.0, 5.0, 6.0, NAN], "lower": [4.0] * 4, "upper": [NAN, 7.0, 8.0, 9.0], "sample_truth": [4.0, 4.0, 4.0, 4.5], "censored": [0.4] * 4}
    kind, text = ev.verdict_repeat(rep, 4.0, "Kurz vor der Abholung", 300)
    assert kind == "info"
    assert "4 feste Zufalls-Seeds, Mechanismus „Kurz vor der Abholung“, n = 300" in text
    assert "Median der Läufe bei Kaplan–Meier 5,00 Tage (Spanne 5,00 bis 6,00; wahrer Median 4,00)" in text
    assert "untere Schranke 4,00, obere Schranke 8,00" in text
    assert "in 3 von 4 Läufen über dem wahren Median, in 0 darunter" in text
    assert "in 4 von 4 Läufen zwischen den Schranken" in text
    assert "obere Schranke ist in 1 von 4 Läufen nicht endlich; der Wert oben gilt nur für die übrigen 3" in text
    assert "Kaplan–Meier hat in 1 von 4 Läufen keinen Median" in text


def test_verdict_repeat_ohne_definierte_werte():
    rep = {"seeds": 2, "km": [NAN, NAN], "lower": [3.0, 3.5], "upper": [NAN, NAN], "sample_truth": [4.0, 4.0], "censored": [1.0, 1.0]}
    _, text = ev.verdict_repeat(rep, 4.0, "Kurz vor der Abholung", 30)
    assert "bei Kaplan–Meier nicht schätzbar (Spanne" in text and "Tage Tage" not in text
    assert "in 0 von 2 Läufen über dem wahren Median, in 0 darunter" in text


def test_verdict_phi_drei_zweige_mit_zahlen():
    kind, text = ev.verdict_phi({"status": A.TIP_ALREADY, "phi_star": 0.0}, 4.5, 4.2, 3.5)
    assert kind == "warning" and text == "Der Schluss „Median ≥ 4,5 Tage“ gilt in dieser Stichprobe schon bei φ = 0 nicht: Der Kaplan–Meier-Median ist 4,20 Tage."
    kind, text = ev.verdict_phi({"status": A.TIP_NEVER, "phi_star": NAN}, 3.5, 5.0, 3.6)
    assert kind == "success" and "bleibt bis φ = 1 bestehen" in text and "läge der Median bei 3,60 Tagen" in text
    kind, text = ev.verdict_phi({"status": A.TIP_TIPS, "phi_star": 0.39}, 4.5, 5.84, 3.35)
    assert kind == "warning" and "gilt für φ = 0 (Kaplan–Meier: 5,84 Tage) und kippt bei φ* = 0,39" in text and "schon 39 % der Ausgeschiedenen" in text and "unter 4,5 Tagen" in text
    kind, text = ev.verdict_phi({"status": A.TIP_TIPS, "phi_star": 0.5}, 4.5, NAN, 3.0)
    assert "Kaplan–Meier: nicht schätzbar)" in text


def test_verdict_world_zahlen_und_zweige():
    rows = [{"t": 4, "s_true": 0.5, "s_b": 0.637, "km_a": 0.636, "km_b": 0.62}, {"t": 6, "s_true": 0.326, "s_b": 0.472, "km_a": 0.484, "km_b": 0.464}]
    kind, text = ev.verdict_world(rows, 4.0, 5.61, 0.019, 0.02, same_world=False)
    assert kind == "info"
    assert "t = 4: wahr 0,500, Ersatzwelt 0,637, Kaplan–Meier aus A 0,636, aus B 0,620" in text and "t = 6: wahr 0,326" in text
    assert "Wahrer Median Welt A 4,00 Tage, Welt B 5,61 Tage" in text and "zwischen den Stichproben aus A und B: 0,019; zwischen zwei Stichproben aus A: 0,020" in text
    _, text = ev.verdict_world(rows, 4.0, NAN, 0.019, 0.02, same_world=False)
    assert "Welt B unendlich (die Kurve fällt nie auf 0,5)" in text
    kind, text = ev.verdict_world(rows, 4.0, 4.0, 0.0, 0.0, same_world=True)
    assert "Welt B ist Welt A" in text and "in beiden 4,00 Tage" in text


def test_tabellen_aus_der_messung_an_einer_kleinen_messdatei():
    cell = {"mech": "Kurz vor der Abholung", "f": 0.4, "q": 0.7, "n": 300, "runs": 10, "censored": {"mean": 0.4, "se": 0.01, "undefined": 0},
            "km": {"mean": 5.6, "se": 0.0123, "undefined": 0}, "lower": {"mean": 3.4, "se": 0.01, "undefined": 0},
            "upper": {"mean": 9.8, "se": 0.5, "undefined": 3}, "sample_truth": {"mean": 4.0, "se": 0.0, "undefined": 0},
            "km_above_truth": 10, "km_defined": 10, "inside_sample": 10, "inside_population": 9}
    row = ev.bounds_table({"bounds_cells": [cell]})[0]
    assert row["f"] == "40 %" and row["q"] == "70 %" and row["ausgeschieden"] == "40,0 %" and row["Median Kaplan–Meier"] == "5,60 ± 0,012"
    assert row["obere Schranke"] == "9,80 ± 0,500 (endlich in 7 von 10 Läufen)" and row["Kaplan–Meier über 4,00"] == "10 von 10"
    assert row["wahre Zeiten der Stichprobe in den Schranken"] == "10 von 10" and row["Median 4,00 in den Schranken"] == "9 von 10"
    cell["upper"] = {"mean": None, "se": None, "undefined": 10}
    assert ev.bounds_table({"bounds_cells": [cell]})[0]["obere Schranke"] == "in keinem von 10 Läufen endlich"
    tip = {"f": 0.2, "q": 0.7, "runs": 5, "status": {A.TIP_ALREADY: 2, A.TIP_TIPS: 3, A.TIP_NEVER: 0}, "phi_star": {"mean": 0.29, "se": 0.01},
           "median_at_phi": {k: {"mean": v} for k, v in zip(("0.0", "0.25", "0.5", "0.75", "1.0"), (4.6, 4.3, 4.1, 3.9, 3.7))}}
    trow = ev.tip_table({"tip_cells": [tip]})[0]
    assert trow["Median bei φ = 0 / 0,25 / 0,5 / 0,75 / 1"] == "4,60 / 4,30 / 4,10 / 3,90 / 3,70"
    assert trow["Schluss gilt schon bei φ = 0 nicht"] == "2 von 5" and trow["kippt bei φ*"] == "3 von 5, im Mittel φ* = 0,29 ± 0,010" and trow["kippt nie"] == "0 von 5"
    rrow = ev.raster_table([{"width": 0.5, "event_first": [0.644, 0.481, 0.365], "censor_first": [0.629, 0.464, 0.348]}])[0]
    assert rrow["Rasterbreite (Tage)"] == "0,50" and rrow["Ereignis zuerst, t = 4"] == "0,644" and rrow["Zensierung zuerst, t = 8"] == "0,348"
    assert ev.bounds_table({}) == [] and ev.tip_table({}) == []
