# Aktivitätserkennung aus Smartphone-Sensordaten

Vergleich von Klassifikationsverfahren (k-NN, SVM, Random Forest, MLP) zur Human Activity Recognition auf dem WISDM-Datensatz.

## Projektziel

Wir untersuchen, wie sich verschiedene Klassifikatoren bei der Aktivitätserkennung verhalten, wenn die **Zeitfenstergröße** für die Feature-Extraktion variiert wird.

**Fragestellung:** Wie beeinflusst die Wahl der Fenstergröße die Klassifikationsgenauigkeit der verschiedenen Verfahren? 

## Status

- [X] Datensatz laden und verarbeiten
- [X] Feature-Extraktion implementiert
- [X] Baseline-Vergleich aller Klassifikatoren (Fenstergröße 200)
- [ ] Experimente mit verschiedenen Fenstergrößen
- [ ] Lernkurven erstellen
- [ ] Visualisierungen für Ausarbeitung

## Datensatz

**WISDM Activity Prediction Dataset v1.1**

- Quelle: https://www.cis.fordham.edu/wisdm/dataset.php
- ~1 Mio. Datenpunkte von 36 Probanden
- Accelerometer-Daten (x, y, z) vom Smartphone
- 6 Aktivitäten: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing

## Setup

```bash
# Repository klonen
git clone https://github.com/Knyllahsyhn/classifier
cd classifier

# Dependencies installieren
pip install numpy pandas scikit-learn scipy matplotlib seaborn

# Skript ausführen
python wisdm_classifier.py
```

## Dateistruktur

```
.
├── README.md
├── wisdm_classifier_comparison.py    # Hauptskript
└── WISDM_ar_v1.1/
    ├── WISDM_ar_v1.1_raw.txt         # Rohdaten
    └── readme.txt
```

## Aktuelle Ergebnisse (Baseline)

Fenstergröße: 200 Samples, 50% Overlap

| Klassifikator | Accuracy | Trainingszeit |
| ------------- | -------- | ------------- |
| MLP           | 97.9%    | 3.41s         |
| k-NN (k=5)    | 97.7%    | 0.00s         |
| Random Forest | 97.6%    | 0.96s         |
| SVM (RBF)     | 94.7%    | 0.43s         |

## TODO

1. **Fenstergrößen-Experiment:** Skript anpassen um verschiedene Fenstergrößen zu testen (z.B. 50, 100, 200, 400)
2. **Ergebnisse plotten:** Accuracy vs. Fenstergröße für alle Klassifikatoren
3. **Analyse:** Warum performen manche Klassifikatoren bei bestimmten Fenstergrößen besser?

## Verwendete Features (33 pro Sample)

Für jede Achse (x, y, z):

- Mean, Std, Min, Max, Range
- Median, Skewness, Kurtosis
- RMS, Absolute Differenzensumme

Zusätzlich Magnitude-Features (mag_mean, mag_std, mag_range)

## Quellen

Kwapisz, J.R., Weiss, G.M. and Moore, S.A. (2010). *Activity Recognition using Cell Phone Accelerometers.* Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (at KDD-10), Washington DC.
