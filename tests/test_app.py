"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# ------------------------------------------------------------------------------------------------ Presets, Ecken und die Verdikt-Zeilen der App

import pytest  # noqa: E402

import inz_constants as C  # noqa: E402

KEY = {"mech": "mech_select", "f": "f_slider", "q": "q_slider", "n": "n_slider"}


def _texts(at) -> str:
    return " ".join(el.value for kind in ("success", "info", "warning", "error") for el in getattr(at, kind))


@pytest.mark.parametrize("name, expected", [
    ("Zwei von fünf scheiden aus (f = 40 %)", "Kaplan–Meier schätzt den Median auf 5,84 Tage (wahr 4,00: +45,9 %, also über der Wahrheit)"),
    ("Jeder Fünfte scheidet aus (f = 20 %)", "Kaplan–Meier schätzt den Median auf 4,44 Tage (wahr 4,00: +11,0 %, also über der Wahrheit)"),
    ("Niemand scheidet aus (f = 0)", "In dieser Stichprobe ist niemand vorzeitig ausgeschieden"),
    ("Gegenrichtung: Langlieger (f = 50 %)", "Kaplan–Meier schätzt den Median auf 3,01 Tage (wahr 4,00: −24,7 %, also unter der Wahrheit)"),
])
def test_jedes_preset_zeigt_das_erwartete_verdikt(name, expected):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert expected in _texts(at)
    assert at.metric[0].label == "Ausgeschieden"


@pytest.mark.parametrize("mech, f, q, n", [
    (C.MECH_NEAR, C.F_MAX, C.Q_MIN, C.N_MIN), (C.MECH_NEAR, C.F_MAX, C.Q_MAX, C.N_MAX), (C.MECH_NEAR, C.F_MIN, C.Q_MIN, C.N_MIN),
    (C.MECH_LONG, C.F_MAX, C.Q_MIN, C.N_MIN), (C.MECH_LONG, C.F_MAX, C.Q_MAX, C.N_MAX), (C.MECH_LONG, C.F_MIN, C.Q_MAX, C.N_MAX),
])
def test_eckfaelle_aller_regler_zusammen(mech, f, q, n):
    at = _fresh_app()
    at.sidebar.selectbox(key=KEY["mech"]).set_value(mech)
    at.sidebar.slider(key=KEY["f"]).set_value(f)
    at.sidebar.slider(key=KEY["q"]).set_value(q)
    at.sidebar.slider(key=KEY["n"]).set_value(n)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error


def test_f_hundert_zeigt_keine_abholung_und_keine_stille_zahl():
    at = _fresh_app()
    at.sidebar.slider(key=KEY["f"]).set_value(100)
    at.run(timeout=TIMEOUT)
    assert not at.exception
    text = _texts(at)
    assert "keine einzige Abholung" in text
    assert [m.value for m in at.metric if m.label == "Median: Kaplan–Meier (Tage)"] == ["kein Median"]
    assert [m.value for m in at.metric if m.label == "Obere Schranke (Tage)"] == ["unendlich"]


def test_experimente_stehen_in_der_app():
    at = _fresh_app()
    titles = [s.value for s in at.subheader]
    for n in (1, 2, 3, 4):
        assert any(t.startswith(f"🔬 Experiment {n}:") for t in titles), n
    assert any("Wo die Annahmen enden" in t for t in titles)
    assert "Annahme, die diese Demo prüft: Die Zensierung ist unabhängig vom Ereignis" in " ".join(i.value for i in at.info)
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)


def test_handbeispiel_annahmen_zeigen_die_mediane_von_hand():
    at = _fresh_app()
    for choice, median in (("Kaplan–Meier", "Median 5 Tage"), ("Alle Zensierten gehen sofort ab", "Median 4 Tage"), ("Alle Zensierten bleiben für immer", "Median 5 Tage")):
        at.radio(key="hand_assumption").set_value(choice).run(timeout=TIMEOUT)
        assert not at.exception
        assert median in _texts(at) and choice in _texts(at)


def test_kipppunkt_regler_aendern_die_aussage():
    at = _fresh_app()
    at.slider(key="threshold_slider").set_value(3.0).run(timeout=TIMEOUT)
    assert "bleibt bis φ = 1 bestehen" in _texts(at)                  # Schwelle 3,0 liegt unter der unteren Schranke 3,35
    at.slider(key="threshold_slider").set_value(7.0).run(timeout=TIMEOUT)
    assert "schon bei φ = 0 nicht" in _texts(at)
    at.slider(key="threshold_slider").set_value(4.5).run(timeout=TIMEOUT)
    at.slider(key="phi_slider").set_value(1.0).run(timeout=TIMEOUT)
    assert not at.exception
    assert [m.value for m in at.metric if m.label.startswith("Median bei φ")] == ["3,35"]
    assert [m.value for m in at.metric if m.label == "Schluss gilt bei diesem φ"] == ["nein"]


def test_pdf_wird_erst_nach_dem_klick_erzeugt():
    at = _fresh_app()
    assert "pdf_bytes" not in at.session_state
    at.button(key="make_pdf").click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["pdf_bytes"][:5] == b"%PDF-"
