# Vergleich von Klassifikationsverfahren zur Aktivitätserkennung aus Smartphone-Sensordaten

**Gruppe:** KiBeMu

---

## Fragestellung

Welcher Klassifikator eignet sich "am besten" zur Erkennung menschlicher Aktivitäten aus Smartphone-Accelerometer-Daten?

#### Was wir betrachten
- kurze Vorstellung aller Verfahren
- Accuracy
- F1
- Lernkurven mit Teilen des Datensatzes
- Confusion (welche Aktivitäten werden am häufigsten verwechselt und warum)




---

## Datensatz: WISDM (Fordham University)

- ~1 Mio. Datenpunkte von 36 Probanden
- 6 Aktivitäten: Walking, Jogging, Treppen (auf/ab), Sitzen, Stehen
- Feature-Extraktion aus Zeitfenstern (statistische Merkmale) notwendig

---

## Klassifikatoren im Vergleich

| Verfahren | Ansatz |
|-----------|--------|
| k-Nearest Neighbors | Distanzbasiert |
| SVM | Trennebene mit maximalem Abstand |
| Random Forest | Ensemble aus Entscheidungsbäumen |
| Multilayer Perceptron | Neuronales Netz |

---

## Erste Ergebnisse

| Klassifikator | Accuracy |
|---------------|----------|
| MLP | 97.9% |
| k-NN | 97.7% |
| Random Forest | 97.6% |
| SVM | 94.7% |

---

## Nächste Schritte

- Lernkurven: Verhalten bei reduzierter Trainingssetgröße
- Fehleranalyse: Welche Aktivitäten werden verwechselt?
- Feature Importance: Welche Merkmale sind entscheidend?-
- Analyse Training- und Inferenzzeit: welcher Klassifikator wäre für eine lokale Anwendung auf einem Smartphone geeignet?
- (Hyperparameter-Tuning: Einfluss von Batch-Size etc. auf Performance)
- (Test mit eigenen Daten)
