# Informatives Zensieren – Streamlit-Demo

<!-- Live-Link nach dem Deploy direkt hierher setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-informatives-zensieren-demo.streamlit.app/)** -->

Demo der Reihe „Überlebenszeit“ (zensierte Zeiten bis zum Ereignis) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie geht von der [kaplan-meier-demo](https://github.com/sebastian-hanisch/kaplan-meier-demo) aus
und prüft die **Annahme, die diese Demo und die weiteren Demos der Reihe treffen: Die Zensierung ist unabhängig vom Ereignis.** Hier ist sie absichtlich verletzt, und die Wahrheit ist bekannt. Vehikel ist die **Containerverweilzeit** der
[verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo) (lognormal, Median 4 Tage, σ = 0,9; dort ohne Zensierung).

## Warum dieses Problem

Ein Container im Block wird irgendwann abgeholt. Scheidet er vorher aus der Beobachtung aus, weil er bald abgeholt würde (Umlagerung, verlorenes Etikett), ist von ihm nur „mindestens so lang“ bekannt, und Kaplan–Meier behandelt ihn, als verhielte er sich ab jetzt wie alle, die noch beobachtet werden.
**Kernfrage: Wie weit kann Kaplan–Meier danebenliegen, und was lässt sich aus den Daten allein dazu sagen?** Die Antwort der Demo: Beim Mechanismus „kurz vor der Abholung“ (q = 70 %, n = 300, Mittel über 1000 Läufe) liegt der Kaplan–Meier-Median bei f = 20 % um 15,8 % und bei f = 40 % um 41,0 % über dem wahren Median 4,00 (Tabelle unten); ob die Zensierung abhängig ist,
lässt sich aus (Zeit, Ereignis) allein, ohne Zusatzannahmen, nicht entscheiden: Zu den hier gewählten Mechanismen gibt es eine unabhängige Ersatzwelt mit denselben beobachtbaren Daten (stetige Zeiten, Übertragung von Tsiatis 1975: unser Schluss); man kann **Schranken** angeben oder **benannte Annahmen** durchspielen.

## Befunde und Korrekturen gegenüber dem Plan

1. **Eigene Zahlen mit SplitMix64 und 1000 statt 200 Läufen.** Die Vormessung (`out_uz_v1_bis_v8.txt`, numpy-Zufall, 200 Läufe, Mechanismus „kurz vor der Abholung“, n = 300) nennt für f = 20 % einen Kaplan–Meier-Median von 4,66 mit Schranken 3,74 bis 5,36 und für f = 40 % 5,67 mit 3,48 bis 9,87.
   Mit dem Code dieser Demo (1000 Läufe): 4,63 mit 3,71 bis 5,33 und 5,64 mit 3,46 bis 9,82 (Tabelle unten). Die Unterschiede betragen höchstens etwa das 1,5-Fache des in der Vormessung genannten Standardfehlers; in der Demo und hier gelten die eigenen Zahlen.
2. **Die Schranken schließen die Wahrheit der Stichprobe immer ein, die der Population nicht immer.** Die Plan-Eigenschaft „bei unabhängiger Zensierung liegt die Wahrheit in den Schranken“ gilt für **jede** Zensierung, die vor dem Ereignis liegt, auch für die abhängige, aber nur für die wahren Zeiten **der Stichprobe**
   (Aufzählung in `tests/test_oracle_informatives_zensieren.py`). Der Median der Population (4,00) liegt bei f = 20 % in 891 von 1000 Läufen zwischen den Schranken (Tabelle).
3. **Kipppunkt: zwei Fälle statt einer Zahl.** Die Vormessung (`uz_v20_bis_v23.py`, Messlogik gelesen) zählte Läufe, in denen der Median schon bei φ = 0 unter 4,5 lag, mit φ* = 0 mit und nannte 0,20 ± 0,02 (f = 20 %) und 0,42 ± 0,01 (f = 40 %), je 200 Läufe.
   Diese Demo trennt die Fälle: Bei f = 20 % gilt der Schluss „Median ≥ 4,5“ in 188 von 500 Läufen schon bei φ = 0 nicht; in 311 kippt er bei im Mittel φ* = 0,29 ± 0,011, in 1 Lauf nie. Bei f = 40 % kippt er in 499 von 500 Läufen bei im Mittel φ* = 0,40 ± 0,006, in 1 Lauf gilt er schon bei φ = 0 nicht.
4. **Zweiter Mechanismus, Richtung gemessen.** Der Plan nannte als offen, ob andere Mechanismen die Richtung ändern („vermutlich nicht, nicht gemessen“). Der zweite Mechanismus („Langlieger früh ausgelagert“) ändert sie: Bei f = 50 % und q = 30 % liegt Kaplan–Meier unter dem wahren Median (Tabelle).
   Bei f = 40 % und q = 70 % ist Kaplan–Meier dagegen kaum verschoben (3,97 gegen 3,99 für die wahren Zeiten der Stichproben). Das sind zwei Beispiele, keine Regel über alle Mechanismen.
5. **Zwei Welten stetig statt im Tagesraster.** Die Vormessung (V21b) baute Welt B im Tagesraster mit zwei Millionen Containern. Diese Demo rechnet die Ersatzwelt **stetig** in geschlossener Form (Satz von Tsiatis gilt für stetige, glatte Verteilungen) und zieht Stichproben aus beiden Welten; das Tagesraster ist als eigene Rechnung dabei (Punkt 6).
6. **Restabweichung der Nachrechnung geklärt (Tagesraster).** Die Nachrechnung des Plans gab Kaplan–Meier bei t = 4 / 6 / 8 mit 0,637 / 0,473 / 0,358 (N = 30000, stetig) gegen die Ersatzwelt im Tagesraster 0,652 / 0,490 / 0,373, eine Abweichung von etwa 0,015, Ursache offen.
   Eigene Rechnung: Die **stetige** Ersatzwelt liefert 0,637 / 0,472 / 0,356; Kaplan–Meier aus Stichproben von A (N = 10000, 20 Zufalls-Seeds) liefert 0,636 / 0,472 / 0,358 mit Standardfehlern von 0,001. Das **Tagesraster** mit der Regel „Ereignis zuerst“ liefert bei Rasterbreite 1 Tag **genau** 0,652 / 0,490 / 0,373 (die Zahlen des Plans);
   mit der Regel „Zensierung zuerst“ 0,623 / 0,457 / 0,341. Mit feiner werdendem Raster nähern sich beide Regeln der stetigen Kurve (bei 0,01 Tagen 0,637 / 0,473 / 0,356 und 0,636 / 0,472 / 0,356). Die Restabweichung stammt also aus dem Runden auf Tage (Gleichstände, je nach Regel nach oben oder unten), nicht aus dem Zufall.
   (Gemessen für Mechanismus „kurz vor der Abholung“, f = 40 %, q = 70 %.)
7. **Wahrheit der Population in geschlossener Form.** Wahre Anteile bei t = 4 / 6 / 8: 0,500 / 0,326 / 0,221 (der Plan nannte 0,503 / 0,328 / 0,222 aus einer Simulation).

## Modell

Die Verweilzeit D ist lognormal (Median 4 Tage, σ = 0,9). Ein Anteil f der Container **scheidet vorzeitig aus**: Die Beobachtung endet bei q·D, vor der Abholung (q < 1). Wer ausscheidet, zählt als zensiert. Zwei Mechanismen (Seitenleiste):

- **Kurz vor der Abholung:** jeder Container scheidet unabhängig mit Wahrscheinlichkeit f aus (Standard: f = 40 %, q = 70 %). Ob ein Container ausscheidet, hängt nicht an D; wann er ausscheidet, schon (bei q·D).
- **Langlieger früh ausgelagert:** es scheidet aus, wer länger bleibt als die Schwelle d mit P(D > d) = f; wer ohnehin lange bleibt, wird früh ausgelagert.

**Die zwei Schranken.** Untere Schranke: jeder Ausgeschiedene geht in Wahrheit sofort ab, sein Ereignis liegt bei seiner Zeit des Ausscheidens. Obere Schranke: er bleibt für immer. Beide sind der Anteil der Zeiten, die länger als t dauern: Ŝ_u(t) = #{T_i > t}/n und Ŝ_o(t) = 1 − #{δ_i = 1, T_i ≤ t}/n.
Kaplan–Meier liegt dazwischen. Die obere Schranke des Medians ist unendlich, wenn weniger als die Hälfte der Container eine gesehene Abholung zeigt.

**Die Annahme φ (Experiment 4).** Der Anteil φ der Ausgeschiedenen geht in Wahrheit sofort ab, der Rest bleibt im Risiko (gewichtete Kurve). φ = 0 ist Kaplan–Meier, φ = 1 die untere Schranke. Das ist ein **einfaches Modell** (ein gemeinsamer Anteil für alle Ausgeschiedenen), kein Standard.

**Die Ersatzwelt (Experiment 3, unser Schluss aus Tsiatis 1975).** Aus den beobachtbaren Teildichten von „Abholung gesehen“ und „ausgeschieden“ folgen die Hazards h_E = q_E/S_obs und h_C = q_C/S_obs. Die Welt B mit **unabhängiger** Abholzeit (Überleben S_B = exp(−∫ h_E)) und Ausscheidezeit (S_C = exp(−∫ h_C))
erzeugt dieselben beobachtbaren Daten wie Welt A, hat aber eine andere Wahrheit. Kaplan–Meier schätzt S_B. Alle Formeln stehen im Expander **📐 Mathematische Formulierung** der App.

## Methodik

Vier Experimente in der App: (1) **Schranken von Hand** an sechs Containern mit Brüchen; (2) **Anteil f**: Kaplan–Meier gegen beide Schranken und die Wahrheit, über 30 feste Zufalls-Seeds und als eingefrorene Messung; (3) **Zwei Welten, dieselben Daten**: Stichproben aus A und B, ihre beobachtbaren Anteile und die Kurven, dazu das Tagesraster als Rechnung;
(4) **Kipppunkt**: wie viele der Ausgeschiedenen müssten in Wahrheit sofort abgehen, damit ein Schluss kippt. Der Kern (Kaplan–Meier, Quantil, SplitMix64) ist eine Kopie aus `ueberlebenszeit-planung/kern/uz_core.py` (Stand 2026-11-10, gekürzt, siehe Kopfzeile von `inz_core.py`); Zufall nur über SplitMix64.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Welt wie oben, n = 300, Zufalls-Seeds fest (`tools/measure_informatives_zensieren.py`, Ergebnis `tools/messung_informatives_zensieren.json`, 1000 Läufe je Zeile, Mittel ± Standardfehler über die Läufe, in denen der Wert endlich ist). „Gegen 4,00“ meint den wahren Median der Population, Basis der Prozente ist die Wahrheit.

| Frage | Befund | Belege |
|---|---|---|
| Handbeispiel (2, 3+, 4, 5, 5+, 8) | Median Kaplan–Meier 5, untere Schranke 4, obere Schranke 5 Tage; obere Schranke S = 5/6, 2/3, 1/2, 1/3 bei t = 2, 4, 5, 8 | `test_algorithm.py` (Brüche von Hand) |
| Anteil f, Mechanismus „kurz vor der Abholung“, q = 70 % | f = 0: Kaplan–Meier 3,99 ± 0,008, beide Schranken gleich. f = 20 %: Kaplan–Meier 4,63 ± 0,011 (+15,8 %), Schranken 3,71 und 5,33. f = 40 %: 5,64 ± 0,015 (+41,0 %), Schranken 3,46 und 9,82. f = 60 %: 7,60 ± 0,025 (+89,9 %), untere Schranke 3,22, obere Schranke in keinem Lauf endlich | Messung (`bounds_cells`), `test_claims.py` |
| Richtung | Kaplan–Meier liegt über 4,00 in 974 von 1000 Läufen (f = 20 %), in 1000 von 1000 (f = 40 % und 60 %); bei f = 0 in 466 von 1000 (nur Zufall) | Messung, `test_claims.py` |
| Lage der Wahrheit zu den Schranken | Die wahren Zeiten der Stichprobe liegen in jedem der 1000 Läufe jeder der neun Zeilen zwischen den Schranken. Der Median der Population 4,00 liegt zwischen den Schranken in 891 (f = 20 %), 986 (f = 40 %), 999 (f = 60 %) von 1000 Läufen, bei f = 0 in 0 von 1000 | Messung; die Eigenschaft für jede Stichprobe in `test_eigenschaften_informatives_zensieren.py` |
| Zeitpunkt q des Ausscheidens, f = 40 % | Kaplan–Meier 4,66 / 5,18 / 5,64 / 6,03 bei q = 30 / 50 / 70 / 90 %; über 4,00 in 971 / 1000 / 1000 / 1000 von 1000 Läufen. Die obere Schranke (9,82) hängt nicht an q, die untere steigt von 2,51 auf 3,82 | Messung, `test_claims.py` |
| Zweiter Mechanismus „Langlieger“ | f = 40 %, q = 70 %: Kaplan–Meier 3,97 ± 0,008 (über 4,00 in 449 von 1000 Läufen; die wahren Zeiten der Stichproben haben im Mittel den Median 3,99: kaum eine Verschiebung), Schranken 3,75 und 3,99 (obere endlich in 999 von 1000 Läufen). f = 50 %, q = 30 %: Kaplan–Meier 3,05 ± 0,004 (−23,7 %), über 4,00 in 0 von 1000 Läufen, Schranken 2,18 und 3,79 (obere endlich in 534 von 1000 Läufen) | Messung, `test_claims.py` |
| Kipppunkt, Schluss „Median ≥ 4,5“, Mechanismus „kurz vor der Abholung“, q = 70 %, 500 Läufe | f = 40 %: Median bei φ = 0 / 0,25 / 0,5 / 0,75 / 1 im Mittel 5,63 / 4,86 / 4,28 / 3,83 / 3,45; kippt in 499 von 500 Läufen bei im Mittel φ* = 0,40 ± 0,006. f = 20 %: 4,62 / 4,35 / 4,13 / 3,92 / 3,72; Schluss gilt schon bei φ = 0 nicht in 188 von 500 Läufen, kippt in 311 bei im Mittel φ* = 0,29 ± 0,011, kippt nie in 1 | Messung (`tip_cells`), `test_claims.py` |
| Zwei Welten, f = 40 %, q = 70 % (N = 10000 je Stichprobe, 20 Zufalls-Seeds) | Wahr (Welt A) bei t = 4 / 6 / 8: 0,500 / 0,326 / 0,221; Ersatzwelt B (Rechnung) 0,637 / 0,472 / 0,356; Kaplan–Meier aus Daten von A 0,636 / 0,472 / 0,358, aus Daten von B 0,636 / 0,473 / 0,358 (Standardfehler 0,001). Wahrer Median A 4,00, B 5,61 Tage. Größter Unterschied der beobachtbaren Anteile (Mittel über die 20 Zufalls-Seeds): A gegen B 0,0106, zwei Stichproben aus A 0,0119 | Messung (`worlds`), `test_oracle_informatives_zensieren.py`, `test_claims.py` |
| Zwei Welten, Mechanismus „Langlieger“, f = 50 %, q = 30 % | Wahr 0,500 / 0,326 / 0,221; Ersatzwelt 0,241 (konstant ab t = 4, weil hinter der Schwelle keine Abholung mehr gesehen wird); Kaplan–Meier aus A 0,241, aus B 0,243. Wahrer Median B 3,06 | Messung, `test_claims.py` |
| Tagesraster (Rechnung ohne Zufall, f = 40 %, q = 70 %, t = 4 / 6 / 8) | Rasterbreite 1 Tag: „Ereignis zuerst“ 0,652 / 0,490 / 0,373, „Zensierung zuerst“ 0,623 / 0,457 / 0,341; 0,5 Tage: 0,644 / 0,481 / 0,365 und 0,629 / 0,464 / 0,348; 0,01 Tage: 0,637 / 0,473 / 0,356 und 0,636 / 0,472 / 0,356; stetig 0,637 / 0,472 / 0,356 | Messung (`rasters`), Neuberechnung und Simulation in `test_oracle_informatives_zensieren.py`, `test_claims.py` |

**Zahl → Test.** Bänder und exakte Werte: Brüche (Handbeispiel, diskrete Zwei-Welten-Rechnung) sind exakt getestet; Zahlen aus Zufall stehen als eingefrorene Messung in der Datei und werden in `test_claims.py` gegen die Neuberechnung einzelner Zeilen (Band über 4 Standardfehler) und gegen den Wortlaut dieser README geprüft.
Die Zahlen der Vormessung (Befunde 1 und 3, 6) stehen in Dateien außerhalb dieses Repos und sind hier nur Vergleich, kein Test.

## Ehrliche Grenzen

- **Zwei gewählte Mechanismen, eine Verteilungsfamilie.** Alle Zahlen gelten für lognormale Verweilzeiten (Median 4, σ = 0,9), n = 300 (Messung) und die gemessenen Zeilen. Der Mechanismus „kurz vor der Abholung“ ist ein Beispiel; „zu lang geschätzt“ gilt in dieser Demo für ihn, nicht für jeden Mechanismus (der zweite Mechanismus kehrt die Richtung bei f = 50 % und q = 30 % um).
- **Tsiatis behandelt Todesursachen, nicht Zensierung.** Der Satz (Tsiatis 1975, PNAS 72(1), 20–22, im Original gelesen, Satz 2, S. 21): Zu jeder glatten gemeinsamen Verteilung der latenten Zeiten gibt es eine mit unabhängigen Zeiten, die dieselben beobachtbaren Teilüberlebensfunktionen liefert.
  Dass das Ausscheiden aus der Beobachtung als zweite „Ursache“ gelesen werden darf und Kaplan–Meier dann die Ersatzwelt schätzt, ist **unser Schluss**; die Rechnung dieser Demo belegt ihn für die zwei gewählten Mechanismen (Zahlen oben), nicht allgemein. Der Satz gilt für stetige, glatte Verteilungen; das Tagesraster liegt außerhalb. Kovariaten kommen bei Tsiatis nicht vor.
- **Mit Zusatzannahmen kann sich Unabhängigkeit erkennen lassen** (Kopplungsfamilie; Heckman und Honoré 1989, nicht gelesen). „Aus den Daten allein nicht prüfbar“ meint Daten der Form (Zeit, Ereignis) ohne weitere Annahmen.
- **Schranken sind weit, wenn viele ausscheiden.** Sie sagen nicht, wo zwischen ihnen die Wahrheit liegt (Peterson 1976, nicht gelesen, nur als Hinweis auf Schranken für gemeinsame Verteilungen genannt).
- **φ ist ein gemeinsamer Anteil für alle Ausgeschiedenen;** der Kipppunkt gilt für diese Annahme und das Raster 0,01, nicht für beliebige Abhängigkeiten.
- **Gegenstück:** Candès, Lei und Ren (JRSS B 2023, im Volltext gelesen) schließen informative Zensierung ausdrücklich aus; ihre Überdeckungszusage gilt nur bei unabhängiger Zensierung. Diese Demo ist nicht ihre Methode.
- **Kein Fremdpaket-Vergleich.** Kaplan–Meier ist selbst geschrieben und gegen Bruchrechnung, Zählung und Aufzählung geprüft, nicht gegen `lifelines`. Die App ist im Browser nicht bei Telefonbreite und im dunklen Design geprüft.

## Tests

144 Tests (`pytest tests/`), davon:

- **Kern** (`tests/test_core.py`): Handbeispiel mit Brüchen, ohne Zensierung gleich eins minus Verteilungsfunktion, Zensierung hinter dem letzten Ereignis, alle zensiert, Eingabeprüfung, SplitMix64 (Referenzwert von Vigna).
- **Orakel** (`tests/test_oracle_informatives_zensieren.py`): Schranken gegen Bruch-Zählung (200 Zufallsinstanzen mit Gleichständen) und gegen die Aufzählung aller möglichen wahren Zeiten (beide Extreme werden angenommen); gewichtete Kurve gegen Bruchrechnung und gegen Vervielfältigung der Zeilen; Kipppunkt gegen das Raster über Vervielfältigung;
  Wahrheit gegen `scipy.stats.lognorm`; Teilverteilungen gegen Simulation (200000 Container) und Quadratur; Ersatzwelt gegen Quadratur der Hazards und gegen Stichproben beider Welten; die diskrete Fassung in Brüchen (40 zufällige abhängige Welten, eine unabhängige Welt, ein Handbeispiel); Tagesraster gegen Kaplan–Meier auf aufgerundeten Zeiten.
- **Eigenschaften** (`tests/test_eigenschaften_informatives_zensieren.py`): Kaplan–Meier zwischen den Schranken an jeder Stelle, Ordnung der Mediane, ohne Ausscheiden fallen alle drei zusammen, obere Schranke endlich genau bei mindestens halb so vielen Abholungen, Skalierung der Zeit, Umsortieren, Median fällt mit φ, mehr Ausscheidende weiten die Schranken, wahre Zeiten liegen in den Schranken.
- **Einheiten, Szenario, Randfälle, Verdikte** (`test_algorithm.py`, `test_scenario.py`, `test_edge_cases.py`, `test_verdicts.py`): Mini-Instanz (sechs Container), Status statt stiller Zahlen (`keine_ereignisse`, `nicht_schaetzbar`, Kipppunkt-Zweige), jedes Verdikt mit jeder Zahl im Text.
- **Aussagen** (`tests/test_claims.py`): README-Zahlen gegen die eingefrorene Messung, Neuberechnung einzelner Zeilen, Testzahl.
- **Oberfläche** (`test_app.py`, `test_layout.py`, `test_presets.py`, `test_pdf.py`, `test_footer.py`, `test_runtime_imports.py`): Rauchtest jedes Buttons und Reglers an den Grenzen, Ecken aller Regler, Verdikt je Preset, Schnellstart mit Hervorhebung, Footer, Laufzeit-Importe gegen `requirements.txt`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: vier Experimente, Ergebnis, Kernfrage, Grenzen, Mathematik |
| `inz_core.py` | Kopie aus `ueberlebenszeit-planung/kern/uz_core.py` (gekürzt): Kaplan–Meier, Quantil, Eingabeprüfung, SplitMix64 |
| `inz_scenario.py` | Welt (lognormal), Erzeuger beider Mechanismen, Wahrheit in geschlossener Form, Handbeispiel |
| `inz_algorithm.py` | Schranken, gewichtete Kurve, Median unter φ, Kipppunkt |
| `inz_ersatz.py` | Teilverteilungen in geschlossener Form, Ersatzwelt, Stichprobe aus Welt B, Tagesraster |
| `inz_evaluation.py` | Einstellungen, Analyse, Handbeispiel in Brüchen, Verdikte als reine Funktionen, Tabellen aus der Messung |
| `inz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `inz_constants.py`, `inz_presets.py`, `inz_layout.py`, `inz_pdf_export.py` | Konstanten und Schnellstart-Szenarien, Presets und Permalink, fester Kopf, PDF |
| `tools/measure_informatives_zensieren.py`, `tools/messung_informatives_zensieren.json` | Messskript (feste Zufalls-Seeds) und sein eingefrorenes Ergebnis |
| `tests/` | siehe Abschnitt „Tests“ |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Laufzeit (ungepinnt), Entwicklung (pytest, scipy für die Orakel), CI |

## Bewusst nicht umgesetzt

- **Schranken für die gemeinsame Verteilung** nach Peterson (1976) und **Kopplungsfamilien** (Heckman und Honoré 1989): beide Quellen sind nicht gelesen.
- **Konfidenzbänder für die Schranken** und Unsicherheit des Kipppunkts je Stichprobe: Die Demo zeigt Spannen über feste Zufalls-Seeds, keine Intervalle.
- **Kovariaten, Gruppen, Gewichtung nach Zensierwahrscheinlichkeit:** nicht umgesetzt.
- **Ein Mechanismus aus echten Daten** oder weitere Mechanismen: nicht gemessen; die zwei Mechanismen sind gewählte Beispiele.
- **Der Mechanismus als kausales Diagramm** (Linie der kausalen Diagramme): nicht umgesetzt und nicht verlinkt.
- **Konforme Grenzen** (Candès, Lei, Ren): nur als Gegenstück genannt; die Methode setzt unabhängige Zensierung voraus und ist nicht Teil der Demo.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die in der Streamlit Cloud eingestellte Python-Version ist nicht geprüft; die CI läuft mit Python 3.12. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Messung neu erzeugen: `python tools/measure_informatives_zensieren.py` (schreibt `tools/messung_informatives_zensieren.json`; die Dauer liegt bei einigen Minuten, hängt vom Rechner ab).

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
