# Experimenteller Aufbau

## Datensätze

Wir verwenden zwei separate WISDM-Datensätze der Fordham University:

- **WISDM AR** (Activity Recognition): ~1 Mio. Rohdatenpunkte von 36 Probanden, wird für das Training verwendet
- **WISDM AT** (Actitracker): ~3 Mio. Rohdatenpunkte von anderen Probanden mit anderen Aufnahmebedingungen, wird für den Test verwendet

Diese Trennung entspricht einem realistischen Szenario: Das Modell wird auf einer Gruppe von Personen trainiert und muss auf völlig unbekannten Personen funktionieren (Cross-Dataset-Evaluation).

## Feature-Extraktion

Die Rohdaten (Accelerometer x, y, z bei ~20 Hz) werden mittels Sliding-Window-Verfahren segmentiert. Für jedes Zeitfenster extrahieren wir 33 statistische Features:

- **Pro Achse (x, y, z):** Mean, Std, Min, Max, Range, Median, Skewness, Kurtosis, RMS, Absolute Differenzensumme
- **Magnitude-Features:** Mean, Std, Range der Vektormagnitude

Die Fenstergröße variiert zwischen 50 und 400 Samples, was bei 20 Hz Samplerate Reaktionszeiten von 2,5 bis 20 Sekunden entspricht. Wir verwenden 50% Overlap zwischen aufeinanderfolgenden Fenstern.

## Klassifikatoren

Wir vergleichen vier Verfahren:

| Verfahren | Beschreibung |
|-----------|--------------|
| **k-Nearest Neighbors (k=5)** | Distanzbasierte Klassifikation anhand der nächsten Nachbarn im Merkmalsraum |
| **Support Vector Machine (RBF-Kernel)** | Sucht eine optimale Trennhyperebene im (transformierten) Merkmalsraum |
| **Random Forest (100 Bäume)** | Ensemble aus Entscheidungsbäumen mit Bagging und Feature-Randomisierung |
| **Multilayer Perceptron (128-64 Neuronen)** | Feedforward-Netz mit zwei versteckten Schichten |

## Experiment A: Variable Trainingssetgröße

Für jede Fenstergröße werden alle verfügbaren Trainingssamples aus dem AR-Datensatz verwendet. Da kleinere Fenster bei gleichem Overlap mehr Samples erzeugen, variiert die Trainingssetgröße:

| Fenstergröße | Reaktionszeit | Trainingssamples (ca.) |
|--------------|---------------|------------------------|
| 50 | 2,5s | ~40.000 |
| 100 | 5,0s | ~20.000 |
| 200 | 10,0s | ~10.000 |
| 400 | 20,0s | ~5.000 |

Test erfolgt auf dem AT-Datensatz.

## Experiment B: Konstante Trainingssetgröße

Um den Effekt der Fenstergröße isoliert zu betrachten, wird die Trainingssetgröße für alle Fenstergrößen auf die Anzahl der größten Fenstergröße reduziert (ca. 5.000 Samples). Die Reduktion erfolgt durch stratifiziertes Sampling, um die Klassenverteilung beizubehalten.

Test erfolgt ebenfalls auf dem AT-Datensatz.

## Vergleich der Experimente

| | Experiment A | Experiment B |
|---|--------------|--------------|
| **Training** | AR (alle Samples) | AR (reduziert auf fixe Anzahl) |
| **Test** | AT | AT |
| **Fragestellung** | Welche Fenstergröße liefert in der Praxis das beste Modell? | Wie wirkt sich die Fenstergröße bei kontrollierter Datenmenge aus? |

Der Vergleich zeigt, ob bessere Performance bei kleinen Fenstern auf mehr Trainingsdaten oder auf den höheren Informationsgehalt kürzerer Zeitfenster zurückzuführen ist.

## Evaluationsmetriken

- **Accuracy:** Anteil korrekt klassifizierter Samples
- **F1-Score (gewichtet):** Harmonisches Mittel aus Precision und Recall, gewichtet nach Klassenhäufigkeit
- **Trainingszeit:** Zeit für Modelltraining
- **Confusion Matrix:** Analyse welche Aktivitäten verwechselt werden
