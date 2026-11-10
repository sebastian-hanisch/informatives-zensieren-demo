"""Informatives Zensieren - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Kaplan–Meier liest zensierte Zeiten richtig, wenn das Ausscheiden aus der Beobachtung nichts mit der noch ausstehenden Zeit zu tun hat. Die Demo zeigt an erfundenen
Containerdaten (die Wahrheit ist bekannt), wie weit Kaplan–Meier danebenliegt, wenn Container ausscheiden, weil sie bald abgeholt würden, und warum sich das aus den Daten
allein nicht erkennen lässt. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import inz_algorithm as A
import inz_constants as C
import inz_core as K
import inz_evaluation as E
from inz_evaluation import Settings, de
from inz_layout import dunkles_design, kopf_css
from inz_pdf_export import build_pdf
from inz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from inz_visualization import build_bounds, build_hand, build_observable, build_phi, build_worlds

st.set_page_config(page_title="Informatives Zensieren – Sebastian Hanisch", layout="wide")


def dl(x: float) -> str:
    """Differenz für `st.metric(delta=...)`: mit ASCII-Minus, denn nur dann zeigt Streamlit den Pfeil nach unten."""
    return E.signed(x).replace("−", "-")


@st.cache_data(show_spinner=False)
def _analysis(settings: Settings):
    return E.analyse(settings)


@st.cache_data(show_spinner=False)
def _repeat(settings: Settings):
    return E.repeat_summary(settings)


@st.cache_data(show_spinner=False)
def _world(settings: Settings, n_world: int):
    return E.world_summary(settings, n_world, settings.seed)


@st.cache_data(show_spinner=False)
def _tip(settings: Settings, threshold: float):
    a = E.analyse(settings)
    return A.tipping_point(a.sample.obs, a.sample.event, threshold)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in inz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🕵️ Informatives Zensieren")
st.markdown(
    """
Ein Container im Block wird irgendwann abgeholt; die **Verweilzeit** ist die Zeit bis dahin. Scheidet ein Container vorher aus der Beobachtung aus, **ist von ihm nur bekannt: „mindestens so lang“**.
**Kaplan–Meier** behandelt ihn dann so, als verhielte er sich ab jetzt wie alle, die noch beobachtet werden. Das ist nur gesichert, wenn das Ausscheiden nichts darüber verrät, **wie bald** er abgeholt würde; sonst kann die Schätzung verschoben sein (Größe und Richtung hängen am Mechanismus).
Scheidet ein Container gerade aus, weil er bald abgeholt wird (Umlagerung, verlorenes Etikett), schätzt Kaplan–Meier die Verweilzeit in dieser Demo zu lang. Aus den Daten allein (Zeit und Ereignis, ohne Zusatzannahmen) lässt sich das **nicht entscheiden**: Zu den hier gewählten Mechanismen gibt es eine unabhängige Ersatzwelt mit denselben beobachtbaren Daten (stetige Zeiten, unser Schluss aus Tsiatis 1975).
Man kann **Schranken** angeben oder **benannte Annahmen** durchspielen, wie weit die Schätzung danebenliegen kann.

Die Daten sind **erfunden** (Simulation, die Wahrheit ist bekannt). Der Fall ohne Ausscheiden steht in der [kaplan-meier-demo](https://github.com/sebastian-hanisch/kaplan-meier-demo), die Containerverweilzeit in der [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo).
Weiter unten: vier **🔬 Experimente** (Schranken von Hand, Anteil f, zwei Welten mit denselben Daten, Kipppunkt), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Demo", expanded=True):
    st.markdown(
        """
1. **Die Welt.** Jeder Container hat eine wahre Verweilzeit D (lognormal, Median 4 Tage). Ein Anteil f der Container **scheidet vorzeitig aus**: Seine Beobachtung endet bei q·D, bevor er abgeholt worden wäre. Er zählt dann als zensiert.
   Der Mechanismus (Seitenleiste) ist ein **gewähltes Beispiel**: „kurz vor der Abholung“ trifft zufällig gewählte Container, „Langlieger früh ausgelagert“ die mit den längsten Verweilzeiten.
2. **Kaplan–Meier** (die Kurve des Anteils, der nach t Tagen noch da ist) lässt jeden Ausgeschiedenen bis zu seinem Ausscheiden mitzählen und danach weg. Eine Zeit bis zur Abholung wird für ihn nicht erfunden.
3. **Zwei Schranken** behandeln die Ausgeschiedenen extrem: **untere Schranke**, alle gehen in Wahrheit sofort ab (ihr Ereignis liegt bei der Zeit des Ausscheidens); **obere Schranke**, alle bleiben für immer.
   Weil in dieser Demo jeder Ausgeschiedene vor seiner Abholung geht, liegen die wahren Zeiten einer Stichprobe immer zwischen den beiden Kurven. Wo Kaplan–Meier zwischen ihnen liegt und wo die Wahrheit, sagen die Schranken nicht.
4. **Was sich nicht prüfen lässt** (Experiment 3): Zu einer abhängigen Welt lässt sich eine **unabhängige Ersatzwelt** mit denselben beobachtbaren Daten, aber einer anderen Wahrheit angeben (mit den dort genannten Grenzen). Kaplan–Meier schätzt die Ersatzwelt.
5. **Was man tun kann:** Schranken berechnen (Experimente 1 und 2) oder den Anteil φ der Ausgeschiedenen, die in Wahrheit sofort abgehen, als **benannte Annahme** durchspielen und den Kipppunkt eines Schlusses suchen (Experiment 4).
        """
    )

st.info(
    "**Annahme, die diese Demo prüft: Die Zensierung ist unabhängig vom Ereignis.** Die anderen Demos dieser Reihe treffen sie (das Ausscheiden verrät nichts über die Restzeit); hier ist sie **verletzt**, absichtlich "
    "und mit bekannter Wahrheit. In echten Daten ist sie eine Annahme, die aus (Zeit, Ereignis) allein, ohne Zusatzannahmen, nicht zu prüfen ist (Experiment 3)."
)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Ausscheiden aus der Beobachtung**")
    mech = st.selectbox("Mechanismus", C.MECHANISMS, key="mech_select", help="\n\n".join(f"**{k}:** {v}" for k, v in C.MECH_HELP.items()))
    f_pct = st.slider("Anteil f der Ausscheidenden (%)", *bounds("f_slider"), step=5, key="f_slider",
                      help="Kurz vor der Abholung: Anteil aller Container, die zufällig ausscheiden. Langlieger: Anteil der Population mit den längsten Verweilzeiten, die ausgelagert wird.")
    q_pct = st.slider("Ausscheiden bei q · D (%)", *bounds("q_slider"), step=5, key="q_slider",
                      help="Wer ausscheidet, verlässt die Beobachtung bei q mal seiner wahren Verweilzeit D, also früher, je kleiner q ist. Nahe 100 %: kurz vor der Abholung.")
    st.markdown("**Stichprobe**")
    n = st.slider("Zahl der Container n", *bounds("n_slider"), key="n_slider", help="Wie viele Container beobachtet werden (die Wahrheit bleibt gleich).")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"mech_select": mech, "f_slider": int(f_pct), "q_slider": int(q_pct), "n_slider": int(n), "seed_input": int(seed)})

settings = Settings(mech, int(f_pct), int(q_pct), int(n), int(seed))
try:
    a = _analysis(settings)
except (K.SurvivalError, ValueError) as err:
    st.error(f"Die Stichprobe konnte nicht ausgewertet werden: {err}")
    st.stop()

# --- Experiment 1: Schranken von Hand ----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Schranken von Hand, sechs Container")
st.markdown(
    "Beobachtete Zeiten in Tagen: **2, 3+, 4, 5, 5+, 8** (ein **+** heißt: vorzeitig ausgeschieden, mindestens so lange da). Wählen Sie, was mit den zwei Ausgeschiedenen **in Wahrheit** geschah; "
    "die gewählte Kurve ist dick, die beiden anderen gepunktet."
)
assumption = st.radio("Annahme über die Ausgeschiedenen", E.HAND_ASSUMPTIONS, horizontal=True, key="hand_assumption")
hand_rows = E.hand_rows(assumption)
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(build_hand(assumption), width="stretch", key="hand_chart")
with c2:
    kind_h, text_h = E.verdict_hand(assumption, hand_rows, E.hand_median(hand_rows))
    getattr(st, kind_h)(text_h)
    st.dataframe(
        [{"t": r["t"], "n": r["n"], "d": r["d"], "(n − d)/n": E.fraction_text(r["factor"]), "S(t)": E.fraction_text(r["s"]), "dezimal": de(float(r["s"]), 4)} for r in hand_rows],
        hide_index=True, width="stretch",
    )
    st.caption("n: im Risiko, d: abgeholt, S(t): Produkt der Faktoren bis t. Bei t = 5 steht eine Abholung neben einer Ausscheidung: Das Ereignis zählt zuerst.")
med_km, med_lo, med_up = (E.hand_median(E.hand_rows(x)) for x in E.HAND_ASSUMPTIONS)
st.caption(f"Mediane: Kaplan–Meier {med_km}, untere Schranke {med_lo}, obere Schranke {med_up} Tage. Die Schranken sind hier eng, weil nur zwei von sechs Containern ausgeschieden sind (Rechenübung, keine Aussage über die Welt).")

st.markdown("---")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: Kaplan–Meier zwischen zwei Schranken")
b = a.bounds
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ausgeschieden", f"{de(100 * a.sample.censored_share, 1)} %", help=f"Anteil der Container, die vorzeitig ausgeschieden sind; {a.sample.n_events} von {a.sample.n} Abholungen gesehen.")
if b.km.status == A.STATUS_OK:
    m2.metric("Median: Kaplan–Meier (Tage)", de(b.km.median), delta=f"{dl(b.km.median - a.true_median)} gegen Wahrheit", delta_color="off", help=f"Wahrer Median der Population: {de(a.true_median)} Tage.")
else:
    m2.metric("Median: Kaplan–Meier (Tage)", "kein Median", delta="Kurve erreicht 0,5 nicht" if b.km.status == A.STATUS_NOT_ESTIMABLE else "keine Abholung gesehen", delta_color="off")
m3.metric("Untere Schranke (Tage)", de(b.lower.median), help="Alle Ausgeschiedenen gehen in Wahrheit sofort ab.")
m4.metric("Obere Schranke (Tage)", de(b.upper.median) if b.upper.status == A.STATUS_OK else "unendlich",
          help="Alle Ausgeschiedenen bleiben für immer. Unendlich: Dann fällt die Kurve nicht auf 0,5.")
st.caption(f"Wahrheit in dieser Welt: Median der Population {de(a.true_median)} Tage; Median der wahren Zeiten dieser Stichprobe {de(a.sample_truth)} Tage (in echten Daten unbekannt). Die Zahl unter dem Wert ist die Abweichung von der Population.")
st.plotly_chart(build_bounds(a), width="stretch", key="bounds_chart")
st.caption("Gepunktet: die wahre Kurve der Population. Gestrichelt: die wahren Zeiten dieser Stichprobe. Senkrechte Striche auf der Kaplan–Meier-Kurve: Ausgeschiedene (höchstens 120 eingezeichnet). Gestrichelt waagerecht: S = 0,5, dort liegt der Median.")

if st.button("📄 PDF erstellen", key="make_pdf", help="Fasst die aktuelle Stichprobe und die Läufe über 30 feste Zufalls-Seeds zusammen."):
    st.session_state["pdf_bytes"] = build_pdf(a, _repeat(settings))
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="informatives-zensieren.pdf", mime="application/pdf", key="download_pdf")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie weit kann Kaplan–Meier danebenliegen?")
kind_v, text_v = E.verdict_core(a)
getattr(st, kind_v)(text_v)

st.markdown("---")

# --- Experiment 2: Anteil f -----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Anteil f und Schranken über viele Stichproben")
st.markdown(
    "Ändern Sie in der Seitenleiste den Anteil f, das q oder den Mechanismus. Eine einzelne Stichprobe ist kein Befund: Darum zeigt die Tabelle die **30 festen Zufalls-Seeds** zu den aktuellen Einstellungen, darunter die eingefrorene Messung mit 1000 Läufen je Zeile."
)
rep = _repeat(settings)
rows_rep = []
for name, key in (("Kaplan–Meier", "km"), ("untere Schranke", "lower"), ("obere Schranke", "upper"), ("wahre Zeiten der Stichprobe", "sample_truth")):
    sm = E.spread(rep[key])
    rows_rep.append({"Größe": name, "Median der Läufe": de(sm["median"]), "Spanne": f"{de(sm['low'])} bis {de(sm['high'])}" if sm["defined"] else "-", "Läufe mit Wert": f"{sm['defined']} von {sm['of']}"})
st.dataframe(rows_rep, hide_index=True, width="stretch")
kind_w, text_w = E.verdict_repeat(rep, a.true_median, settings.mech, settings.n)
getattr(st, kind_w)(text_w)

meas = E.measurements()
if meas.get("bounds_cells"):
    st.markdown(f"**Eingefrorene Messung** (n = {meas['meta']['n']}, {meas['meta']['runs']} Läufe je Zeile, Mittel mit Standardfehler über die Läufe, in denen der Wert endlich ist; wahrer Median der Population {de(meas['meta']['median_true'])})")
    st.dataframe(E.bounds_table(meas), hide_index=True, width="stretch")
    st.caption("„Wahre Zeiten der Stichprobe in den Schranken“ gilt in jedem Lauf (die Schranken sind so gebaut). Der Median der Population (4,00) liegt dagegen nicht immer zwischen den Schranken: "
               "Die Stichprobe weicht vom Median der Population ab. Die Mechanismen sind gewählte Beispiele; andere Mechanismen sind nicht gemessen.")
else:
    st.warning("Die eingefrorene Messung (tools/messung_informatives_zensieren.json) fehlt: Die Tabelle dieses Abschnitts entfällt.")

st.markdown("---")

# --- Experiment 3: Zwei Welten --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Zwei Welten, dieselben Daten")
st.markdown(
    "**Welt A** ist die Welt der Seitenleiste: Das Ausscheiden hängt an der Restzeit (**abhängig**). **Welt B** ist eine **unabhängige Ersatzwelt**: Abholzeit und Ausscheidezeit werden unabhängig gezogen, "
    "mit genau den Hazards, die sich aus den beobachtbaren Daten von A ablesen lassen. Beide Welten erzeugen **dieselben beobachtbaren Daten**, haben aber **verschiedene Wahrheiten**."
)
world_n = st.slider("Container je Welt", C.WORLD_N_MIN, C.WORLD_N_MAX, C.DEFAULT_WORLD_N, key="world_n_slider", help="Je größer, desto kleiner das Rauschen; die Rechenzeit wächst mit n.")
w = _world(settings, int(world_n))
same_world = float(max(abs(w["world"].s_true - w["world"].s_b))) < 1e-4
kind_x, text_x = E.verdict_world(E.world_rows(w), w["median_a"], w["median_b"], w["diff_ab"], w["diff_aa"], same_world)
getattr(st, kind_x)(text_x)
d1, d2 = st.columns(2)
with d1:
    st.markdown("**Was man sieht:** beobachtbare Anteile")
    st.plotly_chart(build_observable(w), width="stretch", key="observable_chart")
    st.caption("Durchgezogen: Stichprobe aus Welt A, gepunktet: Stichprobe aus Welt B. Grün: Abholung gesehen, rot: ausgeschieden. Für die gemessenen Einstellungen unterscheiden sich die Linien nicht stärker als zwei Stichproben aus Welt A (Zahlen in der Messung).")
with d2:
    st.markdown("**Was wahr ist:** Überlebenskurven")
    st.plotly_chart(build_worlds(w), width="stretch", key="worlds_chart")
    st.caption("Kaplan–Meier folgt der Ersatzwelt B, aus Daten von A wie aus Daten von B; die wahre Kurve von A liegt davon entfernt.")
st.info(
    "**Was gezeigt ist und was nicht.** Tsiatis (1975, im Original gelesen, Satz 2, S. 21) zeigt für konkurrierende Todesursachen: Zu jeder glatten gemeinsamen Verteilung der latenten Zeiten gibt es eine mit unabhängigen Zeiten, "
    "die dieselben beobachtbaren Teilüberlebensfunktionen liefert. Dass dasselbe für das Ausscheiden aus der Beobachtung gilt, wenn man es als zweite „Ursache“ neben der Abholung liest, und dass Kaplan–Meier dann die Ersatzwelt schätzt, "
    "ist **unser Schluss**, nicht Tsiatis' Aussage. Der Satz gilt für stetige, glatte Verteilungen; die Rechnung hier ist stetig, das Tagesraster unten liegt außerhalb. Mit Zusatzannahmen (etwa einer Kopplungsfamilie) kann sich Unabhängigkeit "
    "wieder erkennen lassen (Heckman und Honoré 1989, nicht gelesen). Das Gegenstück: Candès, Lei und Ren (2023, im Volltext gelesen) schließen informative Zensierung ausdrücklich aus; ihre Überdeckungszusage gilt nur bei unabhängiger Zensierung."
)
if meas.get("worlds"):
    st.markdown(f"**Eingefrorene Messung: Kaplan–Meier gegen Ersatzwelt** ({meas['worlds'][0]['n_world']} Container je Stichprobe, {meas['worlds'][0]['seeds']} feste Zufalls-Seeds, Mittel ± Standardfehler über die Seeds)")
    rows_w = []
    for cell in meas["worlds"]:
        for i, t in enumerate(cell["times"]):
            rows_w.append({"Mechanismus": cell["mech"], "f / q": f"{round(100 * cell['f'])} % / {round(100 * cell['q'])} %", "t": str(t), "wahr (Welt A)": de(cell["s_true"][i], 3), "Ersatzwelt B (Rechnung)": de(cell["s_b"][i], 3),
                           "Kaplan–Meier aus A": f"{de(cell['km_a']['mean'][i], 3)} ± {de(cell['km_a']['se'][i], 3)}", "Kaplan–Meier aus B": f"{de(cell['km_b']['mean'][i], 3)} ± {de(cell['km_b']['se'][i], 3)}"})
    st.dataframe(rows_w, hide_index=True, width="stretch")
st.markdown("**Tagesraster gegen stetig** (Rechnung ohne Zufall, aktuelle Einstellungen): Wer die Zeiten auf Tage rundet, ändert die Ersatzwelt, weil beim Gleichstand im selben Tag eine Regel gelten muss.")
times_r = list(C.WORLD_TIMES)
st.dataframe(E.raster_table(E.raster_rows(settings.mech, settings.f, settings.q)), hide_index=True, width="stretch")
st.caption(
    "„Ereignis zuerst“: Wer im selben Tag ausscheidet, bleibt für die Abholungen dieses Tages im Risiko (die Regel von Kaplan–Meier). „Zensierung zuerst“: er scheidet vorher aus dem Risiko. "
    f"Mit feiner werdendem Raster nähern sich beide Regeln der stetigen Ersatzwelt ({', '.join(de(w['world'].at('s_b', t), 3) for t in times_r)} bei t = {', '.join(str(t) for t in times_r)})."
)

st.markdown("---")

# --- Experiment 4: Kipppunkt ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Kipppunkt eines Schlusses")
st.markdown(
    "Die Frage: **Wie viele der Ausgeschiedenen müssten in Wahrheit sofort abgegangen sein, damit ein Schluss kippt?** Der Anteil **φ** ist eine **benannte Annahme**, keine Schätzung aus den Daten: "
    "Der Anteil φ jedes Ausgeschiedenen geht bei seiner Zeit des Ausscheidens ab, der Rest bleibt im Risiko. φ = 0 ist Kaplan–Meier, φ = 1 die untere Schranke. "
    "Das Modell (ein gemeinsamer Anteil φ für alle Ausgeschiedenen) ist ein einfaches Beispiel, kein Standard."
)
t1, t2 = st.columns(2)
threshold = t1.slider("Schluss: Der Median ist mindestens … Tage", C.THRESHOLD_MIN, C.THRESHOLD_MAX, C.DEFAULT_THRESHOLD, step=0.1, key="threshold_slider",
                      help="Zum Beispiel eine Zusage oder ein Planwert, der nur gilt, wenn der Median nicht unter dieser Schwelle liegt.")
phi = t2.slider("Annahme φ", 0.0, 1.0, 0.0, step=C.PHI_STEP, key="phi_slider", help="Anteil der Ausgeschiedenen, die in Wahrheit sofort abgehen (0 = Kaplan–Meier, 1 = untere Schranke).")
tip = _tip(settings, float(threshold))
med_phi = A.median_given_phi(a.sample.obs, a.sample.event, float(phi))
r1, r2 = st.columns(2)
r1.metric(f"Median bei φ = {de(phi)} (Tage)", de(med_phi), help="Median der Kurve unter der Annahme φ; nicht schätzbar, wenn die Kurve 0,5 nicht erreicht.")
holds = math.isnan(med_phi) or med_phi >= float(threshold) - 1e-12
r2.metric("Schluss gilt bei diesem φ", "ja" if holds else "nein")
st.plotly_chart(build_phi(tip, float(threshold), float(phi)), width="stretch", key="phi_chart")
kind_p, text_p = E.verdict_phi(tip, float(threshold), b.km.median, b.lower.median)
getattr(st, kind_p)(text_p)
if meas.get("tip_cells"):
    st.markdown(f"**Eingefrorene Messung: Kipppunkt des Schlusses „Median ≥ {de(meas['meta']['tip_threshold'], 1)} Tage“** (Mechanismus „{C.MECH_NEAR}“, q = 70 %, n = {meas['meta']['n']}, {meas['meta']['tip_runs']} Läufe je Zeile, Raster 0,01)")
    st.dataframe(E.tip_table(meas), hide_index=True, width="stretch")
    low = next((c for c in meas["tip_cells"] if c["f"] == 0.2), None)
    note = f" Bei f = 20 % liegt der Kaplan–Meier-Median in {low['status'][A.TIP_ALREADY]} von {low['runs']} Läufen schon bei φ = 0 unter der Schwelle." if low else ""
    st.caption("Das Mittel von φ* gilt nur für die Läufe, in denen der Schluss bei φ = 0 noch besteht." + note)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Zensierung ist unabhängig vom Ereignis** | Scheiden Container wegen ihrer Restzeit aus, schätzt Kaplan–Meier die Ersatzwelt, nicht die Wahrheit (Experiment 3). Ob das vorliegt, zeigen (Zeit, Ereignis) allein, ohne Zusatzannahmen, nicht; die Ersatzwelt dazu ist für stetige Verteilungen gebaut (unser Schluss aus Tsiatis 1975). | Schranken (Experimente 1 und 2), benannte Annahmen wie φ (Experiment 4), Wissen über die Gründe des Ausscheidens |
| **Das Ausscheiden liegt vor dem Ereignis** | Die Schranken sind so gebaut: Ein zensierter Container hat sein Ereignis erst nach der Zensierzeit. Gilt das nicht (etwa verspätete Meldung), gelten die Schranken nicht. | Datenprüfung |
| **Der Mechanismus der Demo ist typisch** | Er ist ein gewähltes Beispiel. Die Richtung der Verzerrung hängt am Mechanismus: Bei „kurz vor der Abholung“ liegt Kaplan–Meier über der Wahrheit, beim zweiten Mechanismus unter ihr (Experiment 2). Andere Mechanismen sind nicht gemessen. | Eigene Annahme über den Mechanismus |
| **Stetige Zeiten** | Der Satz von Tsiatis gilt für stetige, glatte Verteilungen. Das Tagesraster ändert die Ersatzwelt (Experiment 3, Tabelle). | - |
| **Schranken sind brauchbar** | Mit vielen Ausgeschiedenen werden sie weit; ist mehr als die Hälfte ausgeschieden, ist die obere Schranke des Medians unendlich. | Mehr Beobachtung, Zusatzwissen |
| **Eine Welt: lognormal, Median 4, σ = 0,9, keine Merkmale** | Alle Zahlen gelten nur für diese Welt, n = 300 und die gemessenen Zellen; Gruppen, Kovariaten und andere Verteilungen sind nicht gemessen. | - |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen**
- Die Verweilzeit ist lognormal (Median 4 Tage, σ = 0,9); die Daten sind erfunden, kein Echtdatenbezug.
- Ausgeschiedene verlassen die Beobachtung vor ihrer Abholung (bei q · D mit q < 1). Außer dem Ausscheiden gibt es keine Zensierung (kein Beobachtungsfenster).
- Beide Mechanismen sind gewählte Beispiele; keine Zahl gilt für „informatives Zensieren“ im Allgemeinen.

**Grenzen**
- Zwei Mechanismen, eine Verteilungsfamilie, ein gemeinsamer Anteil φ für alle Ausgeschiedenen (Experiment 4). Keine Merkmale, keine Gruppen, keine Intervalle.
- Tsiatis 1975 behandelt Todesursachen, nicht Zensierung: Die Übertragung auf das Ausscheiden ist unser Schluss. Heckman/Honoré 1989 und Peterson 1976 (Schranken) sind nicht gelesen und nur als Hinweis genannt.
- Der Kaplan–Meier-Code ist selbst geschrieben und gegen eine exakte Bruchrechnung und gegen eine Zählung (Schranken) geprüft, nicht gegen ein Fremdpaket (`lifelines` ist nicht Teil des Test-Setups).

**Bekannte Probleme und Unsicherheiten**
- Die Läufe über 30 feste Zufalls-Seeds zeigen eine Spanne, keine Konfidenzgrenzen; die eingefrorene Messung nennt den Standardfehler des Mittels.
- Die Plotly-Diagramme sind nicht mit allen Geräten und Designs getestet.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind für $n$ Container die beobachteten Zeiten $T_i$ und Kennzeichen $\delta_i$ (1: Abholung gesehen, 0: vorzeitig ausgeschieden). Gesucht ist die Überlebensfunktion $S(t)$ der wahren Verweilzeit $D$
und ihr Median. In der Simulation sind auch die wahren Zeiten $D_i$ bekannt; echte Daten haben sie nicht.

| Symbol | Bedeutung |
|---|---|
| $D_i$, $C_i$ | wahre Verweilzeit und Zeit des Ausscheidens von Container $i$ ($C_i = q D_i$ für Ausgeschiedene) |
| $T_i$, $\delta_i$ | Beobachtungszeit $T_i = \min(D_i, C_i)$ und Kennzeichen ($\delta_i = 1$ genau dann, wenn $D_i \le C_i$) |
| $f$, $q$ | Anteil der Ausscheidenden, Zeitpunkt des Ausscheidens als Anteil von $D$ |
| $\hat S$, $\hat S_{\mathrm{u}}$, $\hat S_{\mathrm{o}}$ | Kaplan–Meier, untere und obere Schranke |
| $\varphi$ | angenommener Anteil der Ausgeschiedenen, die in Wahrheit sofort abgehen |
| $q_E$, $q_C$, $S_{\mathrm{obs}}$ | Teildichten von Abholung gesehen und Ausscheiden, Überlebensfunktion der beobachteten Zeit |
| $S_B$, $S_{C}$ | Netto-Überlebensfunktionen der Ersatzwelt (Abholung, Ausscheiden) |

**Kaplan–Meier** mit $t_1 < \dots < t_m$ den Zeiten mit mindestens einer gesehenen Abholung, $n_j = \#\{i : T_i \ge t_j\}$ im Risiko, $d_j$ Abholungen bei $t_j$:

$$\hat S(t) = \prod_{t_j \le t} \left(1 - \frac{d_j}{n_j}\right)$$

**Schranken.** Untere Schranke: jeder Ausgeschiedene geht bei seiner Zeit ab; obere Schranke: er bleibt für immer. Beide sind der Anteil der Zeiten, die länger als $t$ dauern:

$$\hat S_{\mathrm u}(t) = \frac{\#\{i : T_i > t\}}{n}, \qquad \hat S_{\mathrm o}(t) = 1 - \frac{\#\{i : \delta_i = 1,\ T_i \le t\}}{n}$$

Für jede Stichprobe gilt $\hat S_{\mathrm u}(t) \le \hat S(t) \le \hat S_{\mathrm o}(t)$ und, wenn $C_i \le D_i$ für alle Ausgeschiedenen gilt, auch $\hat S_{\mathrm u}(t) \le \frac1n \#\{i : D_i > t\} \le \hat S_{\mathrm o}(t)$.
Die Mediane folgen daraus (kleinste Zeit mit $S \le 0{,}5$; die obere Schranke erreicht 0,5 nur, wenn mindestens die Hälfte der Container eine Abholung zeigt).

**Annahme φ.** Jeder Ausgeschiedene wird in zwei Zeilen geteilt: Gewicht $\varphi$ als Ereignis bei $T_i$, Gewicht $1-\varphi$ zensiert. Kaplan–Meier mit Gewichten $w$:

$$\hat S_\varphi(t) = \prod_{t_j \le t} \left(1 - \frac{\sum_{i:\,T_i = t_j,\ \text{Ereignis}} w_i}{\sum_{i:\,T_i \ge t_j} w_i}\right)$$

$\varphi = 0$ ist Kaplan–Meier, $\varphi = 1$ die untere Schranke. Der Kipppunkt $\varphi^*$ ist das kleinste $\varphi$ des Rasters $0, 0{,}01, \dots, 1$, bei dem der Median unter die Schwelle fällt.

**Die Ersatzwelt (unser Schluss aus Tsiatis 1975, Satz 2).** Aus den beobachtbaren Teildichten werden die Hazards von Abholung und Ausscheiden gebildet,

$$h_E(t) = \frac{q_E(t)}{S_{\mathrm{obs}}(t)}, \qquad h_C(t) = \frac{q_C(t)}{S_{\mathrm{obs}}(t)}, \qquad S_B(t) = e^{-\int_0^t h_E}, \qquad S_C(t) = e^{-\int_0^t h_C}.$$

Dann gilt $S_B \cdot S_C = S_{\mathrm{obs}}$, und die Welt B mit **unabhängiger** Abholzeit (Überleben $S_B$) und Ausscheidezeit (Überleben $S_C$) erzeugt dieselben $q_E$ und $q_C$ wie Welt A. Kaplan–Meier schätzt $S_B$.

**Die Welt dieser Demo.** $D$ lognormal mit Median 4 und $\sigma = 0{,}9$, $S(t) = 1 - \Phi\!\left(\frac{\ln t - \ln 4}{0{,}9}\right)$, Dichte $g$. Mechanismus „kurz vor der Abholung“:

$$S_{\mathrm{obs}}(t) = (1-f)\,S(t) + f\,S(t/q), \qquad q_E(t) = (1-f)\,g(t), \qquad q_C(t) = f\,g(t/q)/q$$

Mechanismus „Langlieger“ mit Schwelle $d$, $S(d) = f$ (ausgelagert wird, wer länger als $d$ bleibt):

$$S_{\mathrm{obs}}(t) = S(\min(t,d)) - \max\!\big(f - S(t/q),\,0\big), \qquad q_E(t) = g(t)\,\mathbf 1(t \le d), \qquad q_C(t) = \tfrac1q\,g(t/q)\,\mathbf 1(t/q > d)$$

**Tagesraster.** Bei Rasterbreite $r$ sei $e_k$ der Anteil gesehener Abholungen und $c_k$ der Anteil Ausgeschiedener im $k$-ten Fenster $((k-1)r, kr]$ und $R_k = S_{\mathrm{obs}}((k-1)r)$ der Anteil im Risiko zu Beginn.
Regel „Ereignis zuerst“: $\hat S = \prod_k (1 - e_k/R_k)$; Regel „Zensierung zuerst“: $\hat S = \prod_k (1 - e_k/(R_k - c_k))$.

**Literatur.** A. A. Tsiatis (1975), A nonidentifiability aspect of the problem of competing risks, Proceedings of the National Academy of Sciences 72(1), 20–22 (im Original gelesen; Satz 2, S. 21).
E. Candès, L. Lei, Z. Ren (2023), Conformalized survival analysis, Journal of the Royal Statistical Society B (arXiv 2103.09763v3, im Volltext gelesen; informative Zensierung ausdrücklich ausgeschlossen).
Nicht gelesen, nur als Hinweis aus Suchergebnissen: A. V. Peterson (1976), Bounds for a joint distribution function with fixed sub-distribution functions, PNAS 73(1); J. J. Heckman, B. E. Honoré (1989), The identifiability of the competing risks model, Biometrika.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
