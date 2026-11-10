"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Zahlen in Beschriftungen werden gerundet (Übertragung klein halten); Zensierungsmarken sind auf höchstens MAX_MARKS ausgedünnt."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import inz_algorithm as A
import inz_constants as C
import inz_core as K
import inz_evaluation as E
import inz_scenario as S

MAX_MARKS = 120
X_CAP = 30.0   # Achse höchstens bis Tag 30: dahinter liegt kaum noch etwas


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def step_xy(km: dict, x_end: float):
    """Treppe als Punktliste für line_shape="hv": von (0, 1) über alle Ereigniszeiten bis x_end."""
    xs = np.concatenate([[0.0], km["t"], [x_end]])
    last = km["s"][-1] if km["s"].size else 1.0
    ys = np.concatenate([[1.0], km["s"], [last]])
    return np.round(xs, 4), np.round(ys, 5)


def x_end(sample: S.Sample) -> float:
    return float(min(sample.obs.max(), X_CAP))


def _thin(values: np.ndarray) -> np.ndarray:
    if values.size <= MAX_MARKS:
        return values
    return values[np.linspace(0, values.size - 1, MAX_MARKS).astype(int)]


def _curve_trace(km: dict, xmax: float, name: str, color: str, width: int = 3, dash: str | None = None):
    xs, ys = step_xy(km, xmax)
    keep = xs <= xmax + 1e-9
    return go.Scatter(x=xs[keep], y=ys[keep], mode="lines", line=dict(color=color, width=width, shape="hv", dash=dash), name=name,
                      hovertemplate="t = %{x:.2f}: S = %{y:.3f}<extra>" + name + "</extra>")


def _axes_survival(fig, xmax: float, **kw):
    fig.update_xaxes(title_text="Tage seit Ankunft", range=[0, xmax * 1.02], **kw)
    fig.update_yaxes(title_text="Anteil noch da, S(t)", range=[0, 1.02], **kw)


def _truth_trace(xmax: float):
    grid = np.linspace(0.0, xmax, 160)
    return go.Scatter(x=np.round(grid, 3), y=np.round(S.true_survival(grid), 5), mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"),
                      name="Wahrheit (lognormal)", hovertemplate="t = %{x:.1f}: S = %{y:.3f}<extra>Wahrheit</extra>")


def _staircase(rows: list[dict], x_stop: float):
    """Treppe eines Handbeispiels als Punktliste (senkrechte Sprünge), bis x_stop verlängert."""
    xs, ys, prev = [0.0], [1.0], 1.0
    for r in rows:
        xs += [float(r["t"]), float(r["t"])]
        ys += [prev, float(r["s"])]
        prev = float(r["s"])
    xs.append(float(x_stop))
    ys.append(prev)
    return xs, ys


def build_hand(assumption: str, times=S.HAND_TIMES, events=S.HAND_EVENTS):
    """Handbeispiel: oben die sechs Lebenslinien, unten die drei Kurven (Kaplan–Meier und beide Schranken), die gewählte hervorgehoben."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.42, 0.58], vertical_spacing=0.08)
    ids = list(range(len(times), 0, -1))
    end = float(max(times)) + 1.0
    forever = assumption == E.HAND_ASSUMPTIONS[2]
    now = assumption == E.HAND_ASSUMPTIONS[1]
    for i, (t, e) in enumerate(zip(times, events)):
        fig.add_trace(go.Scatter(x=[0, t], y=[ids[i]] * 2, mode="lines", line=dict(color="rgba(150,150,150,0.8)", width=3), showlegend=False, hoverinfo="skip"), row=1, col=1)
        if forever and not e:
            fig.add_trace(go.Scatter(x=[t, end], y=[ids[i]] * 2, mode="lines", line=dict(color=C.COLOR_UPPER, width=3, dash="dot"), showlegend=False, hoverinfo="skip"), row=1, col=1)
    seen = [t for t, e in zip(times, events) if e or now]
    fig.add_trace(go.Scatter(x=seen, y=[ids[i] for i, e in enumerate(events) if e or now], mode="markers", marker=dict(symbol="circle", size=11, color=C.COLOR_KM),
                             name="Abholung (gesehen oder angenommen)", hovertemplate="Container %{y}: abgeholt nach %{x} Tagen<extra></extra>"), row=1, col=1)
    cens = [t for t, e in zip(times, events) if not e]
    if not now:
        fig.add_trace(go.Scatter(x=cens, y=[ids[i] for i, e in enumerate(events) if not e], mode="markers", marker=dict(symbol="cross", size=12, color=C.COLOR_LOWER, line=dict(width=2)),
                                 name="ausgeschieden (zensiert)", hovertemplate="Container %{y}: ausgeschieden nach %{x} Tagen<extra></extra>"), row=1, col=1)
    styles = ((E.HAND_ASSUMPTIONS[0], C.COLOR_KM), (E.HAND_ASSUMPTIONS[1], C.COLOR_LOWER), (E.HAND_ASSUMPTIONS[2], C.COLOR_UPPER))
    for name, color in styles:
        xs, ys = _staircase(E.hand_rows(name, times, events), end)
        chosen = name == assumption
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=4 if chosen else 2, dash=None if chosen else "dot"), name=name,
                                 hovertemplate="t = %{x}: S = %{y:.4f}<extra>" + name + "</extra>"), row=2, col=1)
    fig.add_hline(y=0.5, line=dict(color=C.COLOR_TRUTH, width=1, dash="dash"), row=2, col=1)
    fig.update_xaxes(range=[0, end + 0.2], row=2, col=1, title_text="Tage")
    fig.update_yaxes(title_text="Container", tickmode="array", tickvals=ids, ticktext=[str(i) for i in range(1, len(times) + 1)], range=[0.4, len(times) + 0.6], row=1, col=1)
    fig.update_yaxes(title_text="S(t)", range=[0, 1.05], row=2, col=1)
    return _base(fig, height=500)


def _sample_truth_trace(sample: S.Sample, xmax: float):
    grid = np.linspace(0.0, xmax, 200)
    t = np.sort(sample.true_time)
    y = 1.0 - np.searchsorted(t, grid, side="right") / t.size
    return go.Scatter(x=np.round(grid, 3), y=np.round(y, 5), mode="lines", line=dict(color=C.COLOR_TRUTH, width=1.5, dash="dash"), name="wahre Zeiten dieser Stichprobe",
                      hovertemplate="t = %{x:.1f}: S = %{y:.3f}<extra>wahre Zeiten der Stichprobe</extra>")


def build_bounds(a: E.Analysis):
    """Kaplan–Meier, beide Schranken, wahre Kurve (Population) und die wahren Zeiten dieser Stichprobe; die Ausgeschiedenen liegen als Marken auf der Kaplan–Meier-Kurve."""
    xmax = x_end(a.sample)
    fig = go.Figure()
    fig.add_trace(_truth_trace(xmax))
    fig.add_trace(_sample_truth_trace(a.sample, xmax))
    fig.add_trace(_curve_trace(a.bounds.upper.km, xmax, "obere Schranke: alle bleiben für immer", C.COLOR_UPPER, 2))
    fig.add_trace(_curve_trace(a.bounds.lower.km, xmax, "untere Schranke: alle gehen sofort ab", C.COLOR_LOWER, 2))
    fig.add_trace(_curve_trace(a.bounds.km.km, xmax, "Kaplan–Meier", C.COLOR_KM, 3))
    cens = np.sort(a.sample.obs[~a.sample.event])
    cens = _thin(cens[cens <= xmax])
    if cens.size:
        fig.add_trace(go.Scatter(x=np.round(cens, 4), y=np.round(A.survival_values(a.bounds.km.km, cens), 5), mode="markers", marker=dict(symbol="line-ns-open", size=9, color=C.COLOR_KM),
                                 name="ausgeschieden (zensiert)", hovertemplate="ausgeschieden nach %{x:.2f} Tagen<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dash"))
    _axes_survival(fig, xmax)
    return _base(fig, height=420)


def build_phi(tip: dict, threshold: float, phi: float):
    """Median der Kurve gegen die Annahme φ (Anteil der Ausgeschiedenen, die in Wahrheit sofort abgehen), mit der Schwelle des Schlusses."""
    phis, meds = tip["phis"], tip["medians"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(phis, 3), y=np.round(np.where(np.isnan(meds), np.nan, meds), 4), mode="lines", line=dict(color=C.COLOR_KM, width=3), name="Median",
                             hovertemplate="φ = %{x:.2f}: Median %{y:.2f} Tage<extra></extra>", connectgaps=False))
    fig.add_hline(y=threshold, line=dict(color=C.COLOR_LOWER, width=2, dash="dash"), annotation_text=f"Schwelle {E.de(threshold, 1)} Tage", annotation_position="top right")
    fig.add_hline(y=C.MEDIAN_TRUE, line=dict(color=C.COLOR_TRUTH, width=1, dash="dot"), annotation_text="wahrer Median 4", annotation_position="bottom right")
    if not np.isnan(tip["phi_star"]):
        fig.add_vline(x=tip["phi_star"], line=dict(color=C.COLOR_LOWER, width=1, dash="dash"))
    i = int(np.clip(round(phi * (len(phis) - 1)), 0, len(phis) - 1))
    if not np.isnan(meds[i]):
        fig.add_trace(go.Scatter(x=[phis[i]], y=[meds[i]], mode="markers", marker=dict(size=13, color=C.COLOR_LINE, line=dict(width=2, color="white")), name="gewähltes φ",
                                 hovertemplate="φ = %{x:.2f}: Median %{y:.2f} Tage<extra></extra>"))
    fig.update_xaxes(title_text="φ: Anteil der Ausgeschiedenen, die in Wahrheit sofort abgehen", range=[0, 1])
    fig.update_yaxes(title_text="Median (Tage)")
    return _base(fig, height=340)


def build_observable(w: dict):
    """Was man sieht: die beobachtbaren Anteile (Abholung gesehen, ausgeschieden) in Stichproben aus Welt A und Welt B; die beiden Welten liegen übereinander."""
    grid = w["grid"]
    fig = go.Figure()
    for (ev, ce), name, dash, width in ((w["cif_a"], "Welt A (abhängig)", None, 3), (w["cif_b"], "Welt B (unabhängig)", "dot", 3)):
        color_e, color_c = (C.COLOR_KM, C.COLOR_LOWER)
        fig.add_trace(go.Scatter(x=grid, y=np.round(ev, 4), mode="lines", line=dict(color=color_e, width=width, dash=dash, shape="hv"), name=f"Abholung gesehen, {name}",
                                 hovertemplate="t = %{x:.1f}: %{y:.3f}<extra>Abholung gesehen</extra>"))
        fig.add_trace(go.Scatter(x=grid, y=np.round(ce, 4), mode="lines", line=dict(color=color_c, width=width, dash=dash, shape="hv"), name=f"ausgeschieden, {name}",
                                 hovertemplate="t = %{x:.1f}: %{y:.3f}<extra>ausgeschieden</extra>"))
    fig.update_xaxes(title_text="Tage seit Ankunft", range=[0, 20])
    fig.update_yaxes(title_text="Anteil aller Container bis t", range=[0, 1.02])
    return _base(fig, height=360)


def _grid_trace(km: dict, xs: np.ndarray, name: str, color: str, width: int, dash: str | None = None):
    """Kaplan–Meier-Kurve auf einem festen Raster (für große Stichproben: nur 321 Punkte statt jeder Ereigniszeit)."""
    return go.Scatter(x=xs, y=np.round(A.survival_values(km, xs), 5), mode="lines", line=dict(color=color, width=width, shape="hv", dash=dash), name=name,
                      hovertemplate="t = %{x:.2f}: S = %{y:.3f}<extra>" + name + "</extra>")


def build_worlds(w: dict):
    """Was wahr ist: die wahre Kurve von Welt A, die Ersatzwelt B (geschlossene Form) und Kaplan–Meier aus den Daten beider Welten (alle Kurven auf einem Raster von 321 Punkten)."""
    world = w["world"]
    xmax = 20.0
    xs = np.round(np.linspace(0.0, xmax, 321), 4)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=np.round(np.interp(xs, world.t, world.s_true), 5), mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="wahre Kurve (Welt A)",
                             hovertemplate="t = %{x:.1f}: S = %{y:.3f}<extra>Welt A, wahr</extra>"))
    fig.add_trace(go.Scatter(x=xs, y=np.round(np.interp(xs, world.t, world.s_b), 5), mode="lines", line=dict(color=C.COLOR_WORLD_B, width=2, dash="dash"), name="Ersatzwelt B (geschlossene Form)",
                             hovertemplate="t = %{x:.1f}: S = %{y:.3f}<extra>Welt B, wahr</extra>"))
    fig.add_trace(_grid_trace(w["km_a"].km, xs, "Kaplan–Meier aus den Daten von A", C.COLOR_KM, 3))
    fig.add_trace(_grid_trace(w["km_b"].km, xs, "Kaplan–Meier aus den Daten von B", C.COLOR_WORLD_B, 2, "dot"))
    fig.add_hline(y=0.5, line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dash"))
    _axes_survival(fig, xmax)
    return _base(fig, height=400)
