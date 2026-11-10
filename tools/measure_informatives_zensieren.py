"""Messung der Demo „Informatives Zensieren“ (feste Zufalls-Seeds, Zufall nur über SplitMix64); schreibt tools/messung_informatives_zensieren.json.

Aufruf (aus dem Repo-Ordner):  <python> tools/measure_informatives_zensieren.py [--runs 1000]
Die Laufzeit liegt bei einigen Minuten (Größenordnung; hängt vom Rechner ab). Die Datei wird von der App (Tabellen) und von tests/test_claims.py gelesen.

Blöcke:
  bounds_cells   Kaplan–Meier und beide Schranken über viele Stichproben (n = 300) je Mechanismus, f, q: Mittel und Standardfehler der Mediane,
                 Zahl der Läufe ohne Median, Lage der Wahrheit der Stichprobe und der Population zu den Schranken.
  tip_cells      Kipppunkt φ* des Schlusses „Median ≥ 4,5“ (Experiment 4).
  worlds         Zwei Welten, dieselben Daten (Experiment 3): geschlossene Ersatzwelt gegen Kaplan–Meier aus Stichproben beider Welten (N = 10000 je Stichprobe),
                 dazu die beobachtbaren Anteile; Tagesraster gegen stetig (Rechnung ohne Zufall).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import inz_algorithm as A  # noqa: E402
import inz_constants as C  # noqa: E402
import inz_core as K  # noqa: E402
import inz_ersatz as E  # noqa: E402
import inz_evaluation as EV  # noqa: E402
import inz_scenario as S  # noqa: E402

N_SAMPLE = 300
BOUNDS_CELLS = (
    (C.MECH_NEAR, 0.0, 0.7), (C.MECH_NEAR, 0.2, 0.7), (C.MECH_NEAR, 0.4, 0.7), (C.MECH_NEAR, 0.6, 0.7),
    (C.MECH_NEAR, 0.4, 0.3), (C.MECH_NEAR, 0.4, 0.5), (C.MECH_NEAR, 0.4, 0.9),
    (C.MECH_LONG, 0.4, 0.7), (C.MECH_LONG, 0.5, 0.3),
)
TIP_CELLS = ((0.2, 0.7), (0.4, 0.7))
TIP_THRESHOLD = 4.5
WORLD_CELLS = ((C.MECH_NEAR, 0.4, 0.7), (C.MECH_LONG, 0.5, 0.3))
WORLD_N = 10000
WORLD_SEEDS = 20
SEED_BOUNDS, SEED_TIP, SEED_WORLD = 10000, 20000, 30000


def mean_se(values) -> dict:
    """Mittel und Standardfehler der definierten Werte, dazu wie viele fehlen (nan)."""
    a = np.asarray(values, dtype=float)
    ok = a[~np.isnan(a)]
    if ok.size == 0:
        return {"mean": None, "se": None, "undefined": int(a.size)}
    se = float(ok.std(ddof=1) / np.sqrt(ok.size)) if ok.size > 1 else 0.0
    return {"mean": round(float(ok.mean()), 5), "se": round(se, 5), "undefined": int(a.size - ok.size)}


def bounds_cell(mech: str, f: float, q: float, runs: int) -> dict:
    km, lo, up, truth, share = [], [], [], [], []
    inside_sample = inside_pop = above = defined = 0
    for r in range(runs):
        s = S.simulate(mech, f, q, N_SAMPLE, SEED_BOUNDS + r)
        b = A.bounds(s.obs, s.event)
        tr = A.sample_median_true(s.true_time)
        km.append(b.km.median), lo.append(b.lower.median), up.append(b.upper.median), truth.append(tr), share.append(s.censored_share)
        inside_sample += A.inside(b.lower.median, b.upper.median, tr)
        inside_pop += A.inside(b.lower.median, b.upper.median, C.MEDIAN_TRUE)
        if not np.isnan(b.km.median):
            defined += 1
            above += b.km.median > C.MEDIAN_TRUE
    return {
        "mech": mech, "f": f, "q": q, "n": N_SAMPLE, "runs": runs, "censored": mean_se(share),
        "km": mean_se(km), "lower": mean_se(lo), "upper": mean_se(up), "sample_truth": mean_se(truth),
        "km_above_truth": above, "km_defined": defined, "inside_sample": int(inside_sample), "inside_population": int(inside_pop),
    }


def tip_cell(f: float, q: float, runs: int) -> dict:
    stat = {A.TIP_ALREADY: 0, A.TIP_TIPS: 0, A.TIP_NEVER: 0}
    stars, med_at = [], {p: [] for p in (0.0, 0.25, 0.5, 0.75, 1.0)}
    monotone = True
    for r in range(runs):
        s = S.simulate(C.MECH_NEAR, f, q, N_SAMPLE, SEED_TIP + r)
        tip = A.tipping_point(s.obs, s.event, TIP_THRESHOLD)
        stat[tip["status"]] += 1
        if tip["status"] == A.TIP_TIPS:
            stars.append(tip["phi_star"])
        meds = tip["medians"]
        monotone &= bool(np.all(np.diff(np.where(np.isnan(meds), np.inf, meds)) <= 1e-12))
        for p in med_at:
            med_at[p].append(meds[int(round(p * (C.PHI_GRID - 1)))])
    return {"mech": C.MECH_NEAR, "f": f, "q": q, "n": N_SAMPLE, "runs": runs, "threshold": TIP_THRESHOLD, "status": stat, "phi_star": mean_se(stars),
            "median_at_phi": {str(p): mean_se(v) for p, v in med_at.items()}, "medians_nonincreasing_in_phi": bool(monotone)}


def world_cell(mech: str, f: float, q: float) -> dict:
    world = E.ersatz_world(mech, f, q)
    times = list(C.WORLD_TIMES)
    out = {"mech": mech, "f": f, "q": q, "n_world": WORLD_N, "seeds": WORLD_SEEDS, "times": times,
           "s_true": [round(world.at("s_true", t), 5) for t in times], "s_b": [round(world.at("s_b", t), 5) for t in times]}
    km_a, km_b, d_ab, d_aa = [], [], [], []
    grid = np.linspace(0.0, 30.0, 301)
    for r in range(WORLD_SEEDS):
        sa = S.simulate(mech, f, q, WORLD_N, SEED_WORLD + 3 * r)
        sb = E.sample_world_b(world, WORLD_N, SEED_WORLD + 3 * r + 1)
        sa2 = S.simulate(mech, f, q, WORLD_N, SEED_WORLD + 3 * r + 2)
        ka, kb = K.kaplan_meier(sa.obs, sa.event), K.kaplan_meier(sb.obs, sb.event)
        km_a.append([K.survival_at(ka, t) for t in times])
        km_b.append([K.survival_at(kb, t) for t in times])
        ea, ca = E.empirical_cifs(sa, grid)
        eb, cb = E.empirical_cifs(sb, grid)
        ea2, ca2 = E.empirical_cifs(sa2, grid)
        d_ab.append(max(np.abs(ea - eb).max(), np.abs(ca - cb).max()))
        d_aa.append(max(np.abs(ea - ea2).max(), np.abs(ca - ca2).max()))
    for key, arr in (("km_a", km_a), ("km_b", km_b)):
        a = np.asarray(arr)
        out[key] = {"mean": [round(float(x), 5) for x in a.mean(axis=0)], "se": [round(float(x), 5) for x in a.std(axis=0, ddof=1) / np.sqrt(a.shape[0])]}
    out["diff_ab"] = {"mean": round(float(np.mean(d_ab)), 5), "max": round(float(np.max(d_ab)), 5)}
    out["diff_aa"] = {"mean": round(float(np.mean(d_aa)), 5), "max": round(float(np.max(d_aa)), 5)}
    out["median_b"] = round(float(EV.curve_median(world, "s_b")), 4)
    out["rasters"] = [{"width": r["width"], "event_first": [round(x, 5) for x in r["event_first"]], "censor_first": [round(x, 5) for x in r["censor_first"]]}
                      for r in EV.raster_rows(mech, f, q, times)]
    return out




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--tip-runs", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()
    result = {"meta": {"runs": args.runs, "tip_runs": args.tip_runs, "n": N_SAMPLE, "median_true": C.MEDIAN_TRUE, "sigma_true": C.SIGMA_TRUE,
                       "seed_bounds": SEED_BOUNDS, "seed_tip": SEED_TIP, "seed_world": SEED_WORLD, "tip_threshold": TIP_THRESHOLD}}
    result["bounds_cells"] = [bounds_cell(m, f, q, args.runs) for m, f, q in BOUNDS_CELLS]
    print(f"bounds_cells fertig nach {time.time() - t0:.0f} s", flush=True)
    result["tip_cells"] = [tip_cell(f, q, args.tip_runs) for f, q in TIP_CELLS]
    print(f"tip_cells fertig nach {time.time() - t0:.0f} s", flush=True)
    result["worlds"] = [world_cell(m, f, q) for m, f, q in WORLD_CELLS]
    print(f"worlds fertig nach {time.time() - t0:.0f} s", flush=True)
    path = ROOT / C.MEASURE_FILE
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"geschrieben: {path}")


if __name__ == "__main__":
    main()
