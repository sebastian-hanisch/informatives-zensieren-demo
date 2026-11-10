"""Konstanten: Regler-Grenzen, feste Annahmen der Welt, Farben und Presets."""

# --- Die Welt: Verweilzeit D eines Containers im Block, lognormal (wie in der Verweilzeit-Demo) ------------------------------------------------
MEDIAN_TRUE = 4.0            # Tage
SIGMA_TRUE = 0.9             # σ der Log-Skala; ergibt Mittel 5,997 und das 0,9-Quantil 12,7

# --- Mechanismen des Ausscheidens (Auswahl in der Seitenleiste) -------------------------------------------------------------------------------
MECH_NEAR = "Kurz vor der Abholung"
MECH_LONG = "Langlieger früh ausgelagert"
MECHANISMS = (MECH_NEAR, MECH_LONG)
MECH_HELP = {
    MECH_NEAR: "Ein zufälliger Anteil f der Container verlässt die Beobachtung bei q·D (D = Zeit bis zur Abholung), also kurz bevor er abgeholt worden wäre (Umlagerung, verlorenes Etikett).",
    MECH_LONG: "Die Container mit den längsten Verweilzeiten (der Anteil f der Population) verlassen die Beobachtung bei q·D; wer ohnehin lange bleibt, wird früh ausgelagert.",
}

# --- Regler -------------------------------------------------------------------------------------------------------------------------------------
# Prozentregler sind ganzzahlig (Schritt 5), damit die Hervorhebung des Presets nie an Gleitkomma-Gleichheit hängt.
N_MIN, N_MAX, DEFAULT_N = 30, 1000, 300     # Kaplan–Meier ist O(n·m) in dieser Fassung (m Ereigniszeiten); bei n = 1000 bleibt ein Rerun deutlich unter einer Sekunde
F_MIN, F_MAX, DEFAULT_F = 0, 100, 40        # Anteil f in Prozent
Q_MIN, Q_MAX, DEFAULT_Q = 10, 95, 70        # q in Prozent: Ausscheiden bei q·D
SEED_MAX, DEFAULT_SEED = 999999, 2026
PHI_STEP = 0.05                             # Regler φ in Experiment 4: Schritt in Anteilen (0 bis 1)
THRESHOLD_MIN, THRESHOLD_MAX, DEFAULT_THRESHOLD = 3.0, 7.0, 4.5   # Schwelle des Schlusses „Median ≥ Schwelle“ (Experiment 4)
WORLD_N_MIN, WORLD_N_MAX, DEFAULT_WORLD_N = 300, 5000, 2000      # Container je Welt in Experiment 3 (O(n·m): bei 5000 gut eine Zehntelsekunde je Kurve)

# --- Messungen ----------------------------------------------------------------------------------------------------------------------------------
REPEAT_SEEDS = tuple(range(8000, 8030))     # 30 feste Zufalls-Seeds für den Lauf über viele Stichproben (nie eine einzelne Stichprobe als Befund)
PHI_GRID = 101                              # φ-Raster 0, 0,01, …, 1 für den Kipppunkt
MEASURE_FILE = "tools/messung_informatives_zensieren.json"   # eingefrorenes Ergebnis von tools/measure_informatives_zensieren.py
T_GRID_MAX = 1000.0                         # Rechengitter der Ersatzwelt (Tage); dahinter liegt weniger als 1e-9 der Wahrscheinlichkeit (S(1000) = 4,4e-10)
RASTERS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)  # Rasterbreiten (Tage) der Rechnung „Tagesraster gegen stetig“
WORLD_TIMES = (4, 6, 8)                     # Zeitpunkte, an denen Experiment 3 die Kurven vergleicht

# --- Farben -------------------------------------------------------------------------------------------------------------------------------------
COLOR_TRUTH = "#555555"
COLOR_KM = "#2F6B65"
COLOR_LOWER = "#d1495b"      # untere Schranke: alle Zensierten gehen sofort
COLOR_UPPER = "#edae49"      # obere Schranke: alle Zensierten bleiben für immer
COLOR_WORLD_B = "#4c78a8"
COLOR_BAND = "rgba(47,107,101,0.18)"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Zwei von fünf scheiden aus (f = 40 %)": {"mech": MECH_NEAR, "f": 40, "q": 70, "n": DEFAULT_N, "seed": DEFAULT_SEED},
    "Jeder Fünfte scheidet aus (f = 20 %)": {"mech": MECH_NEAR, "f": 20, "q": 70, "n": DEFAULT_N, "seed": DEFAULT_SEED},
    "Niemand scheidet aus (f = 0)": {"mech": MECH_NEAR, "f": 0, "q": 70, "n": DEFAULT_N, "seed": DEFAULT_SEED},
    "Gegenrichtung: Langlieger (f = 50 %)": {"mech": MECH_LONG, "f": 50, "q": 30, "n": DEFAULT_N, "seed": DEFAULT_SEED},
}
# Die Zahlen der Hilfetexte gelten für den Zufalls-Seed 2026 und stehen in tests/test_presets.py gegen die Rechnung.
PRESET_HELP = {
    "Zwei von fünf scheiden aus (f = 40 %)": "300 Container; 40 % verlassen die Beobachtung bei 70 % ihrer Zeit bis zur Abholung. In dieser Stichprobe schätzt Kaplan–Meier den Median auf 5,84 Tage (wahr 4,00), "
                                             "die Schranken liegen bei 3,35 und 10,29 Tagen.",
    "Jeder Fünfte scheidet aus (f = 20 %)": "Dasselbe mit 20 % Ausscheidenden: Kaplan–Meier schätzt 4,44 Tage, die Schranken liegen bei 3,64 und 5,27 Tagen.",
    "Niemand scheidet aus (f = 0)": "Kontrolle: Ohne Ausscheiden fallen Kaplan–Meier und beide Schranken zusammen (3,88 Tage in dieser Stichprobe; wahrer Median der Population 4,00).",
    "Gegenrichtung: Langlieger (f = 50 %)": "Die Hälfte der Container, und zwar die mit den längsten Verweilzeiten, verlässt die Beobachtung bei 30 % ihrer Zeit. Hier schätzt Kaplan–Meier den Median auf 3,01 Tage, "
                                            "unter der Wahrheit 4,00; die Schranken liegen bei 2,15 und 3,88 Tagen. Die Richtung der Verzerrung hängt am Mechanismus.",
}
