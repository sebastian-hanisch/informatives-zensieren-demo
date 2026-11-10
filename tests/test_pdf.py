"""PDF-Export: erzeugt gültige Bytes aus der aktuellen Stichprobe, Umlaute gehen, Zeichen außerhalb der Kernschrift werden ersetzt."""

import inz_constants as C
import inz_evaluation as E
import inz_pdf_export as PDF


def test_pdf_ist_ein_pdf_und_nennt_die_einstellung():
    a = E.analyse(E.Settings())
    data = PDF.build_pdf(a, E.repeat_summary(a.settings))
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_plain_ersetzt_minuszeichen_gedankenstrich_sterne_und_sonderzeichen():
    assert PDF._plain("−21,3 % – **fett** ∞ φ ≥ „x“") == '-21,3 % - fett unendlich phi >= "x"'


def test_pdf_ohne_abholung_und_mit_langlieger_wirft_keinen_fehler():
    for settings in (E.Settings(C.MECH_NEAR, 100, 70, 40, 1), E.Settings(C.MECH_LONG, 50, 30, 300, 2), E.Settings(C.MECH_NEAR, 0, 70, 30, 3)):
        a = E.analyse(settings)
        assert PDF.build_pdf(a, E.repeat_summary(settings))[:5] == b"%PDF-"
