"""Aussagen: Jede Zahl der README steht in der eingefrorenen Messung (tools/messung_informatives_zensieren.json) und wird hier gegen den Wortlaut der README geprüft;
einzelne Zeilen der Messung werden mit dem Code der Demo neu gerechnet (Band über 4 Standardfehler), die Rechnungen ohne Zufall exakt.

Ausnahmen: Die Zahlen der Vormessung (Befunde 1, 3 und 6 der README) stehen in Dateien außerhalb dieses Repos."""

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import inz_algorithm as A
import inz_constants as C
import inz_core as K
import inz_ersatz as ER
import inz_evaluation as E
import inz_scenario as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
MEAS = json.loads((ROOT / C.MEASURE_FILE).read_text(encoding="utf-8"))
CELLS = {(c["mech"], c["f"], c["q"]): c for c in MEAS["bounds_cells"]}
NEAR, LONG = C.MECH_NEAR, C.MECH_LONG


def _measure_module():
    spec = importlib.util.spec_from_file_location("measure_inz", ROOT / "tools" / "measure_informatives_zensieren.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def de(x, nd=2):
    return E.de(x, nd)


def triple(xs, nd=3):
    return " / ".join(de(x, nd) for x in xs)


def _has(text: str):
    assert text in README, text


def test_messdatei_meta_stimmt_mit_den_angaben_der_readme():
    meta = MEAS["meta"]
    assert (meta["runs"], meta["tip_runs"], meta["n"], meta["median_true"], meta["sigma_true"]) == (1000, 500, 300, 4.0, 0.9)
    assert all(c["runs"] == 1000 and c["n"] == 300 for c in MEAS["bounds_cells"]) and all(c["runs"] == 500 for c in MEAS["tip_cells"])
    assert "1000 Läufe je Zeile" in README and "n = 300" in README and "500 Läufe" in README


def test_readme_anteil_f_zeilen_gegen_messung():
    for f in (0.2, 0.4, 0.6):
        c = CELLS[(NEAR, f, 0.7)]
        _has(f"{de(c['km']['mean'])} ± {de(c['km']['se'], 3)} ({E.pct(E.deviation_percent(c['km']['mean'], 4.0))})")
    c2, c4 = CELLS[(NEAR, 0.2, 0.7)], CELLS[(NEAR, 0.4, 0.7)]
    _has(f"bei f = 20 % um {de(E.deviation_percent(c2['km']['mean'], 4.0), 1)} % und bei f = 40 % um {de(E.deviation_percent(c4['km']['mean'], 4.0), 1)} % über dem wahren Median 4,00")
    c0 = CELLS[(NEAR, 0.0, 0.7)]
    _has(f"Kaplan–Meier {de(c0['km']['mean'])} ± {de(c0['km']['se'], 3)}, beide Schranken gleich")
    assert c0["km"] == c0["lower"] == c0["upper"]
    for f in (0.2, 0.4):
        c = CELLS[(NEAR, f, 0.7)]
        _has(f"Schranken {de(c['lower']['mean'])} und {de(c['upper']['mean'])}")
    c6 = CELLS[(NEAR, 0.6, 0.7)]
    _has(f"untere Schranke {de(c6['lower']['mean'])}, obere Schranke in keinem Lauf endlich")
    assert c6["upper"]["mean"] is None and c6["upper"]["undefined"] == 1000


def test_readme_richtung_und_lage_der_wahrheit_gegen_messung():
    above = [CELLS[(NEAR, f, 0.7)]["km_above_truth"] for f in (0.2, 0.4, 0.6)]
    assert above == [974, 1000, 1000]
    _has("974 von 1000 Läufen (f = 20 %), in 1000 von 1000 (f = 40 % und 60 %); bei f = 0 in 466 von 1000")
    assert CELLS[(NEAR, 0.0, 0.7)]["km_above_truth"] == 466
    assert all(c["inside_sample"] == c["runs"] == 1000 for c in MEAS["bounds_cells"]) and len(MEAS["bounds_cells"]) == 9
    inside = [CELLS[(NEAR, f, 0.7)]["inside_population"] for f in (0.2, 0.4, 0.6)]
    assert inside == [891, 986, 999] and CELLS[(NEAR, 0.0, 0.7)]["inside_population"] == 0
    _has("891 (f = 20 %), 986 (f = 40 %), 999 (f = 60 %) von 1000 Läufen, bei f = 0 in 0 von 1000")
    _has("liegt bei f = 20 % in 891 von 1000 Läufen zwischen den Schranken")


def test_readme_q_zeilen_gegen_messung():
    cells = [CELLS[(NEAR, 0.4, q)] for q in (0.3, 0.5, 0.7, 0.9)]
    _has(f"Kaplan–Meier {' / '.join(de(c['km']['mean']) for c in cells)} bei q = 30 / 50 / 70 / 90 %")
    _has(f"über 4,00 in {' / '.join(str(c['km_above_truth']) for c in cells)} von 1000 Läufen")
    uppers = {c["upper"]["mean"] for c in cells}
    assert len(uppers) == 1                                          # die obere Schranke hängt nicht an q
    _has(f"Die obere Schranke ({de(uppers.pop())}) hängt nicht an q, die untere steigt von {de(cells[0]['lower']['mean'])} auf {de(cells[-1]['lower']['mean'])}")
    assert [c["lower"]["mean"] for c in cells] == sorted(c["lower"]["mean"] for c in cells)


def test_readme_zweiter_mechanismus_gegen_messung():
    a, b = CELLS[(LONG, 0.4, 0.7)], CELLS[(LONG, 0.5, 0.3)]
    _has(f"Kaplan–Meier {de(a['km']['mean'])} ± {de(a['km']['se'], 3)} (über 4,00 in {a['km_above_truth']} von 1000 Läufen; die wahren Zeiten der Stichproben haben im Mittel den Median {de(a['sample_truth']['mean'])}")
    _has(f"Schranken {de(a['lower']['mean'])} und {de(a['upper']['mean'])}")
    _has(f"Kaplan–Meier {de(b['km']['mean'])} ± {de(b['km']['se'], 3)} ({E.pct(E.deviation_percent(b['km']['mean'], 4.0))}), über 4,00 in {b['km_above_truth']} von 1000 Läufen, Schranken {de(b['lower']['mean'])} und {de(b['upper']['mean'])}")
    _has(f"obere endlich in {1000 - b['upper']['undefined']} von 1000 Läufen")
    assert b["km_above_truth"] == 0 and b["km"]["mean"] < 4.0 < CELLS[(NEAR, 0.4, 0.7)]["km"]["mean"]           # die Richtung kehrt sich um
    _has(f"(3,97 gegen {de(a['sample_truth']['mean'])} für die wahren Zeiten der Stichproben)")


def test_readme_kipppunkt_gegen_messung():
    t20, t40 = (next(c for c in MEAS["tip_cells"] if c["f"] == f) for f in (0.2, 0.4))
    for t in (t20, t40):
        assert t["medians_nonincreasing_in_phi"] is True and t["threshold"] == 4.5
    keys = ("0.0", "0.25", "0.5", "0.75", "1.0")
    _has(f"f = 40 %: Median bei φ = 0 / 0,25 / 0,5 / 0,75 / 1 im Mittel {' / '.join(de(t40['median_at_phi'][k]['mean']) for k in keys)}; kippt in {t40['status'][A.TIP_TIPS]} von 500 Läufen bei im Mittel φ* = {de(t40['phi_star']['mean'])} ± {de(t40['phi_star']['se'], 3)}")
    _has(f"f = 20 %: {' / '.join(de(t20['median_at_phi'][k]['mean']) for k in keys)}; Schluss gilt schon bei φ = 0 nicht in {t20['status'][A.TIP_ALREADY]} von 500 Läufen, kippt in {t20['status'][A.TIP_TIPS]} bei im Mittel φ* = {de(t20['phi_star']['mean'])} ± {de(t20['phi_star']['se'], 3)}, kippt nie in {t20['status'][A.TIP_NEVER]}")
    assert sum(t20["status"].values()) == sum(t40["status"].values()) == 500
    _has(f"Bei f = 20 % gilt der Schluss „Median ≥ 4,5“ in {t20['status'][A.TIP_ALREADY]} von 500 Läufen schon bei φ = 0 nicht; in {t20['status'][A.TIP_TIPS]} kippt er bei im Mittel φ* = {de(t20['phi_star']['mean'])} ± {de(t20['phi_star']['se'], 3)}, in {t20['status'][A.TIP_NEVER]} Lauf nie")
    _has(f"Bei f = 40 % kippt er in {t40['status'][A.TIP_TIPS]} von 500 Läufen bei im Mittel φ* = {de(t40['phi_star']['mean'])} ± {de(t40['phi_star']['se'], 3)}, in {t40['status'][A.TIP_ALREADY]} Lauf gilt er schon bei φ = 0 nicht")


def test_readme_zwei_welten_gegen_messung():
    near, long_ = MEAS["worlds"]
    assert (near["mech"], long_["mech"]) == (NEAR, LONG) and near["times"] == [4, 6, 8] and near["n_world"] == 10000 and near["seeds"] == 20
    _has(f"Wahr (Welt A) bei t = 4 / 6 / 8: {triple(near['s_true'])}; Ersatzwelt B (Rechnung) {triple(near['s_b'])}; Kaplan–Meier aus Daten von A {triple(near['km_a']['mean'])}, aus Daten von B {triple(near['km_b']['mean'])} (Standardfehler 0,001)")
    assert max(near["km_a"]["se"] + near["km_b"]["se"] + long_["km_a"]["se"] + long_["km_b"]["se"]) < 0.0015 and min(near["km_a"]["se"]) > 0.0005
    _has(f"Wahrer Median A 4,00, B {de(near['median_b'])} Tage")
    _has(f"A gegen B {de(near['diff_ab']['mean'], 4)}, zwei Stichproben aus A {de(near['diff_aa']['mean'], 4)}")
    _has(f"Wahr 0,500 / 0,326 / 0,221; Ersatzwelt {de(long_['s_b'][0], 3)} (konstant ab t = 4")
    _has(f"Kaplan–Meier aus A {de(long_['km_a']['mean'][0], 3)}, aus B {de(long_['km_b']['mean'][0], 3)}. Wahrer Median B {de(long_['median_b'])}")
    assert long_["s_b"][0] == pytest.approx(long_["s_b"][-1], abs=1e-5)                       # hinter der Schwelle fällt die Ersatzkurve nicht mehr
    # die beobachtbaren Anteile unterscheiden sich zwischen A und B nicht stärker als zwischen zwei Stichproben aus A
    for w in (near, long_):
        assert w["diff_ab"]["mean"] < 1.5 * w["diff_aa"]["mean"] and w["diff_ab"]["max"] < 0.03
    _has("Wahre Anteile bei t = 4 / 6 / 8: 0,500 / 0,326 / 0,221")
    assert triple(near["s_true"]) == "0,500 / 0,326 / 0,221"


def test_readme_tagesraster_gegen_messung():
    near = MEAS["worlds"][0]
    by = {r["width"]: r for r in near["rasters"]}
    ef = lambda w: triple(by[w]["event_first"])      # noqa: E731
    cf = lambda w: triple(by[w]["censor_first"])     # noqa: E731
    _has(f"Rasterbreite 1 Tag: „Ereignis zuerst“ {ef(1.0)}, „Zensierung zuerst“ {cf(1.0)}")
    _has(f"0,5 Tage: {ef(0.5)} und {cf(0.5)}; 0,01 Tage: {ef(0.01)} und {cf(0.01)}; stetig {triple(near['s_b'])}")
    _has(f"liefert bei Rasterbreite 1 Tag **genau** {ef(1.0)} (die Zahlen des Plans)")
    _has(f"mit der Regel „Zensierung zuerst“ {cf(1.0)}")
    _has(f"(bei 0,01 Tagen {ef(0.01)} und {cf(0.01)})")
    _has(f"liefert {triple(near['s_b'])}; Kaplan–Meier aus Stichproben von A")
    assert ef(1.0) == "0,652 / 0,490 / 0,373"                      # die Zahlen des Plans (V21b, Tagesraster)
    assert abs(near["rasters"][0]["event_first"][0] - near["s_b"][0]) == pytest.approx(0.0154, abs=0.0005)         # die „Restabweichung von etwa 0,015“


def test_messung_einer_zeile_gegen_neuberechnung():
    mod = _measure_module()
    for (mech, f, q) in ((NEAR, 0.4, 0.7), (LONG, 0.5, 0.3)):
        frozen = CELLS[(mech, f, q)]
        new = mod.bounds_cell(mech, f, q, 100)
        scale = math.sqrt(1000 / 100)
        for key in ("km", "lower", "sample_truth", "censored"):
            assert abs(new[key]["mean"] - frozen[key]["mean"]) < 4 * scale * max(frozen[key]["se"], 1e-3), (mech, key)
        assert new["inside_sample"] == 100
        assert abs(new["km_above_truth"] - frozen["km_above_truth"] / 10) <= 6


def test_messung_kipppunkt_neuberechnung_und_monotonie():
    mod = _measure_module()
    frozen = next(c for c in MEAS["tip_cells"] if c["f"] == 0.4)
    new = mod.tip_cell(0.4, 0.7, 40)
    assert new["medians_nonincreasing_in_phi"] is True
    assert new["status"][A.TIP_TIPS] >= 36 and abs(new["phi_star"]["mean"] - frozen["phi_star"]["mean"]) < 4 * math.sqrt(500 / 40) * frozen["phi_star"]["se"]


def test_messung_zwei_welten_rechnungen_exakt_und_stichproben_im_band():
    mod = _measure_module()
    for frozen in MEAS["worlds"]:
        mech, f, q = frozen["mech"], frozen["f"], frozen["q"]
        w = ER.ersatz_world(mech, f, q)
        assert [round(w.at("s_true", t), 5) for t in frozen["times"]] == frozen["s_true"]
        assert [round(w.at("s_b", t), 5) for t in frozen["times"]] == frozen["s_b"]
        assert [{"width": r["width"], "event_first": [round(x, 5) for x in r["event_first"]], "censor_first": [round(x, 5) for x in r["censor_first"]]} for r in E.raster_rows(mech, f, q, frozen["times"])] == frozen["rasters"]
        # zwei der 20 Stichproben neu: Kaplan–Meier bei t = 4 / 6 / 8 im Band um das eingefrorene Mittel (Standardfehler der Einzelstichprobe = se · √20)
        for r in range(2):
            sa = S.simulate(mech, f, q, mod.WORLD_N, mod.SEED_WORLD + 3 * r)
            km = K.kaplan_meier(sa.obs, sa.event)
            for i, t in enumerate(frozen["times"]):
                assert abs(K.survival_at(km, t) - frozen["km_a"]["mean"][i]) < 4 * math.sqrt(20) * max(frozen["km_a"]["se"][i], 5e-4), (mech, r, t)


def test_vormessungszahlen_stehen_als_vergleich_in_der_readme():
    for text in ("4,66 mit Schranken 3,74 bis 5,36", "5,67 mit 3,48 bis 9,87", "0,20 ± 0,02", "0,42 ± 0,01", "0,637 / 0,473 / 0,358", "0,652 / 0,490 / 0,373", "0,503 / 0,328 / 0,222"):
        _has(text)
    assert "`out_uz_v1_bis_v8.txt`" in README


def test_testzahl_in_der_readme_stimmt_mit_der_suite_ueberein():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=600).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, out[-500:]
    assert f"{m.group(1)} Tests (`pytest tests/`)" in README
