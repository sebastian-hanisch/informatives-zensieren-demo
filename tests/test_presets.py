"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; die Zahlen der Hilfetexte stehen gegen die Rechnung."""

import re

import pytest

import inz_constants as C
import inz_evaluation as ev
import inz_presets as P


def _settings(preset: dict) -> ev.Settings:
    return ev.Settings(preset["mech"], preset["f"], preset["q"], preset["n"], preset["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert preset["mech"] in C.MECHANISMS
        assert C.F_MIN <= preset["f"] <= C.F_MAX and C.Q_MIN <= preset["q"] <= C.Q_MAX and C.N_MIN <= preset["n"] <= C.N_MAX and 0 <= preset["seed"] <= C.SEED_MAX
        assert preset["f"] % 5 == 0 and preset["q"] % 5 == 0          # die Prozentregler haben Schritt 5


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS[next(iter(C.PRESETS))]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_setting_specs_defaults_are_the_default_settings():
    d = ev.Settings()
    assert [P.SETTING_SPECS[k].default for k in ("mech_select", "f_slider", "q_slider", "n_slider", "seed_input")] == [d.mech, d.f_pct, d.q_pct, d.n, d.seed]


def _de(x: float) -> str:
    return ev.de(x)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_zahlen_im_hilfetext_stimmen_mit_der_rechnung_ueberein(name):
    a = ev.analyse(_settings(C.PRESETS[name]))
    text = C.PRESET_HELP[name]
    numbers = re.findall(r"\d+,\d{2}", text)
    b = a.bounds
    have = {_de(b.km.median), _de(b.lower.median), _de(b.upper.median), _de(a.true_median)}
    if name.startswith("Niemand"):
        have |= {_de(a.sample_truth)}
    assert numbers and set(numbers) <= have, (name, numbers, have)


def test_hilfetexte_richtungen_stimmen_mit_den_presets():
    """Die Richtung „zu lang“ gilt bei den beiden Mechanismus-A-Presets mit Ausscheiden, die Gegenrichtung beim Langlieger-Preset, null Ausscheiden ergibt eine Kurve."""
    res = {name: ev.analyse(_settings(p)) for name, p in C.PRESETS.items()}
    five, fifth, none, back = (res[n] for n in C.PRESETS)
    assert five.bounds.km.median > five.true_median and fifth.bounds.km.median > fifth.true_median
    assert abs(five.bounds.km.median - five.true_median) > abs(fifth.bounds.km.median - fifth.true_median)         # 40 % verzerren mehr als 20 %
    assert (five.bounds.upper.median - five.bounds.lower.median) > (fifth.bounds.upper.median - fifth.bounds.lower.median)
    assert none.sample.n_events == none.sample.n and none.bounds.km.median == none.bounds.lower.median == none.bounds.upper.median
    assert back.bounds.km.median < back.true_median
    assert "Richtung der Verzerrung hängt am Mechanismus" in C.PRESET_HELP[next(reversed(C.PRESETS))]
