"""PDF-Zusammenfassung der aktuellen Einstellung (fpdf2, Kernschrift Helvetica: Umlaute gehen, Gedankenstrich, Minuszeichen und € nicht).

Die App erzeugt das PDF erst nach dem Klick auf „PDF erstellen“; hier steht nur der Aufbau aus fertigen Zahlen."""

from fpdf import FPDF

import inz_evaluation as E


def _plain(text: str) -> str:
    """Kernschrift-sicher: Minuszeichen, Gedankenstrich, Unendlich-Zeichen und Grieche werden ersetzt, Markdown-Sterne fallen weg."""
    return (text.replace("−", "-").replace("–", "-").replace("**", "").replace("∞", "unendlich").replace("φ", "phi").replace("≥", ">=")
            .replace("„", '"').replace("“", '"'))


def build_pdf(a: E.Analysis, rep: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def line(text: str, size: int = 10, bold: bool = False, gap: float = 5.5):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.multi_cell(width, gap, _plain(text), new_x="LMARGIN", new_y="NEXT")

    st = a.settings
    line("Informatives Zensieren: Schranken statt Punktschätzer", 15, True, 8)
    line("Sebastian Hanisch - Demo, erfundene Daten (Verweilzeit lognormal, Median 4 Tage, Sigma 0,9)", 9)
    pdf.ln(2)
    line("Einstellung", 11, True)
    line(f"Mechanismus: {st.mech}; Anteil f = {st.f_pct} %; q = {st.q_pct} %; n = {st.n} Container; Zufalls-Seed {st.seed}.")
    line(f"Vorzeitig ausgeschieden: {E.de(100 * a.sample.censored_share, 1)} % von {a.sample.n} Containern ({a.sample.n_events} Abholungen gesehen).")
    pdf.ln(2)
    line("Mediane dieser Stichprobe", 11, True)
    b = a.bounds
    line(f"Kaplan-Meier: {E.de(b.km.median)} Tage; untere Schranke (alle gehen sofort ab): {E.de(b.lower.median)}; obere Schranke (alle bleiben für immer): {E.de(b.upper.median)}.")
    line(f"Wahrheit: Median der Population {E.de(a.true_median)} Tage; Median der wahren Zeiten dieser Stichprobe {E.de(a.sample_truth)} Tage.")
    pdf.ln(2)
    line("Kernfrage", 11, True)
    line(E.verdict_core(a)[1])
    pdf.ln(2)
    line(f"Über {rep['seeds']} feste Zufalls-Seeds", 11, True)
    line(E.verdict_repeat(rep, a.true_median, st.mech, st.n)[1])
    pdf.ln(2)
    line("Grenze: Der Mechanismus des Ausscheidens ist ein gewähltes Beispiel. Aus (Zeit, Ereignis) allein lässt sich ohne Zusatzannahmen nicht erkennen, ob die Zensierung unabhängig ist (stetige Zeiten, zwei gewählte Mechanismen); "
         "alle Zahlen stammen aus der Simulation dieser Demo.", 9)
    return bytes(pdf.output())
