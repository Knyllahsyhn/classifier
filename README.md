# Aktivitätserkennung aus Smartphone-Sensordaten

Vergleich von Klassifikationsverfahren (k-NN, SVM, Random Forest, MLP) zur Human Activity Recognition auf dem WISDM-Datensatz.

## Fragestellung

**Wie beeinflusst die Reaktionszeit (Fenstergröße) die Klassifikationsgenauigkeit bei der Aktivitätserkennung?**

Praktische Motivation: Ein Smartphone soll in Echtzeit erkennen, welche Aktivität der Nutzer ausführt. Je schneller die Erkennung (kleineres Zeitfenster), desto weniger Sensordaten stehen zur Verfügung. Ab welcher Fenstergröße bricht die Genauigkeit ein?


## Datensätze

Wir verwenden zwei WISDM-Datensätze der Fordham University für Cross-Dataset-Evaluation:

| Datensatz | Verwendung | Beschreibung |
|-----------|------------|--------------|
| **WISDM AR v1.1** | Training | ~1 Mio. Datenpunkte, 36 Probanden |
| **WISDM AT v2.0** | Test | ~3 Mio. Datenpunkte, andere Probanden |

- Quelle: https://www.cis.fordham.edu/wisdm/dataset.php
- Sensordaten: Accelerometer (x, y, z) bei ~20 Hz
- 6 Aktivitäten: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing

## Setup

```bash
# Repository klonen
git clone https://github.com/Knyllahsyhn/classifier
cd classifier

# Dependencies installieren
pip install -r requirements.txt

# Jupyter Notebook starten
jupyter notebook wisdm_experiment.ipynb
```

Die Datensätze werden beim ersten Start automatisch heruntergeladen.

## Dateistruktur

```
.
├── README.md
├── requirements.txt
├── wisdm_experiment.ipynb      # Hauptnotebook mit allen Experimenten
├── experimenteller_aufbau.md   # Methodenbeschreibung für Ausarbeitung
├── kurzvorstellung.md          # Präsentationsfolien
└── data/                       # (wird automatisch erstellt)
    ├── WISDM_ar_v1.1/
    └── WISDM_at_v2.0/
```

## Experimente

### Experiment A: Variable Trainingssetgröße

Für jede Fenstergröße werden alle verfügbaren Trainingssamples verwendet. Kleinere Fenster erzeugen mehr Samples.

### Experiment B: Konstante Trainingssetgröße

Trainingssamples werden auf eine fixe Anzahl reduziert (stratifiziertes Sampling), um den Effekt der Fenstergröße isoliert zu betrachten.

| Fenstergröße | Reaktionszeit | Trainingssamples (A) | Trainingssamples (B) |
|--------------|---------------|----------------------|----------------------|
| 50 | 2.5s | ~40.000 | ~5.400 |
| 100 | 5.0s | ~20.000 | ~5.400 |
| 200 | 10.0s | ~10.000 | ~5.400 |
| 400 | 20.0s | ~5.400 | ~5.400 |

Beide Experimente testen auf dem AT-Datensatz (Cross-Dataset-Evaluation).

## Klassifikatoren

| Verfahren | Konfiguration |
|-----------|---------------|
| k-Nearest Neighbors | k=5 |
| Support Vector Machine | RBF-Kernel, C=1.0 |
| Random Forest | 100 Bäume |
| Multilayer Perceptron | 128-64 Neuronen, Early Stopping |

## Features (33 pro Sample)

Für jede Achse (x, y, z):
- Mean, Std, Min, Max, Range
- Median, Skewness, Kurtosis
- RMS, Absolute Differenzensumme

Magnitude-Features:
- mag_mean, mag_std, mag_range

## Generierte Plots

Das Notebook erzeugt folgende Visualisierungen im `output/`-Verzeichnis:

| Datei | Beschreibung |
|-------|--------------|
| `exp_a_accuracy.png` | Accuracy-Kurven Experiment A |
| `exp_a_heatmap.png` | Heatmap Experiment A |
| `exp_b_accuracy.png` | Accuracy-Kurven Experiment B |
| `exp_b_heatmap.png` | Heatmap Experiment B |
| `accuracy_difference.png` | Differenz A - B |
| `comparison_bars.png` | Balkendiagramm A vs B |
| `training_time_comparison.png` | Trainingszeiten |
| `f1_score_comparison.png` | F1-Scores |
| `confusion_matrix.png` | Confusion Matrix (einzeln) |
| `confusion_matrix_comparison.png` | Confusion Matrix A vs B |
| `feature_importance.png` | Top 15 Features |
| `feature_importance_categories.png` | Importance nach Kategorie |
| `feature_importance_heatmap.png` | Importance über Fenstergrößen |

## Quellen

Kwapisz, J.R., Weiss, G.M. and Moore, S.A. (2010). *Activity Recognition using Cell Phone Accelerometers.* Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (at KDD-10), Washington DC.

## Autoren

Gruppe KiBeMu – OTH Regensburg, WS 2025/26
