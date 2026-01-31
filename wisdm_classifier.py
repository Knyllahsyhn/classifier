#!/usr/bin/env python3
"""
WISDM Dataset - Fenstergrößen-Experiment
Untersucht: Wie beeinflusst die Fenstergröße (= Reaktionszeit) die Klassifikationsgenauigkeit?

Trainiert auf: WISDM AR Datensatz
Testet auf: WISDM AT Datensatz (realistischere Daten) - optional
"""

import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURATION
# =============================================================================

# Fenstergrößen zum Testen (bei ~20 Hz Samplerate)
WINDOW_SIZES = [20, 30, 50, 75, 100, 150, 200, 300, 400]
# Entspricht ca.: 1s, 1.5s, 2.5s, 3.75s, 5s, 7.5s, 10s, 15s, 20s

STEP_RATIO = 0.5  # 50% Overlap

# Datensatz-Pfade
TRAIN_DATA_PATH = "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"  # AR für Training
TEST_DATA_PATH = "WISDM_at_v2.0/WISDM_at_v2.0_raw.txt"  # AT für Test

# Auf AT-Datensatz testen
USE_AT_FOR_TESTING = True  # Auf True setzen für Testing auf ActiTracker-Dataset

SAMPLE_RATE_HZ = 20  # Ungefähre Samplerate des Datensatzes


# =============================================================================
# FUNKTIONEN
# =============================================================================

def load_wisdm_data(filepath):
    """Lädt WISDM Rohdaten aus Textdatei"""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().rstrip(';')
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 6:
                try:
                    user = int(parts[0])
                    activity = parts[1].strip()
                    timestamp = int(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5].rstrip(';'))
                    data.append([user, activity, timestamp, x, y, z])
                except (ValueError, IndexError):
                    continue
    return pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])


def extract_features(window):
    """Extrahiert statistische Features aus einem Zeitfenster"""
    features = []
    for axis in ['x', 'y', 'z']:
        values = window[axis].values
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.max(values) - np.min(values),
            np.median(values),
            stats.skew(values),
            stats.kurtosis(values),
            np.sqrt(np.mean(values ** 2)),
            np.sum(np.abs(np.diff(values)))
        ])

    # Magnitude Features
    magnitude = np.sqrt(window['x'] ** 2 + window['y'] ** 2 + window['z'] ** 2)
    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude) - np.min(magnitude)
    ])

    return features


def create_dataset(df, window_size, step_size):
    """Erstellt Feature-Datensatz aus Rohdaten mit gegebener Fenstergröße"""
    X_list = []
    y_list = []

    for (user, activity), group in df.groupby(['user', 'activity']):
        group = group.sort_values('timestamp').reset_index(drop=True)

        for start in range(0, len(group) - window_size + 1, step_size):
            window = group.iloc[start:start + window_size]
            features = extract_features(window)
            X_list.append(features)
            y_list.append(activity)

    X = np.array(X_list)
    y = np.array(y_list)

    # NaN/Inf bereinigen
    mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X = X[mask]
    y = y[mask]

    return X, y


def get_classifiers():
    """Gibt Dictionary mit allen Klassifikatoren zurück"""
    return {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
    }


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

print("=" * 70)
print("WISDM Fenstergrößen-Experiment")
print("Fragestellung: Ab welcher Reaktionszeit wird die Erkennung unzuverlässig?")
print("=" * 70)

# --- Daten laden ---
print("\n[1/3] Lade Trainingsdaten (WISDM AR)...")
df_train = load_wisdm_data(TRAIN_DATA_PATH)
print(f"      {len(df_train):,} Datenpunkte geladen")

if USE_AT_FOR_TESTING:
    print("\n      Lade Testdaten (WISDM AT)...")
    df_test = load_wisdm_data(TEST_DATA_PATH)
    print(f"      {len(df_test):,} Datenpunkte geladen")

# --- Experiment durchführen ---
print("\n[2/3] Starte Experiment...")
print(f"      Fenstergrößen: {WINDOW_SIZES}")
print(f"      Reaktionszeiten: {[f'{w / SAMPLE_RATE_HZ:.1f}s' for w in WINDOW_SIZES]}")

results = []

for window_size in WINDOW_SIZES:
    step_size = int(window_size * STEP_RATIO)
    reaction_time = window_size / SAMPLE_RATE_HZ

    print(f"\n--- Fenstergröße: {window_size} ({reaction_time:.1f}s) ---")

    # Features extrahieren
    X_train_full, y_train_full = create_dataset(df_train, window_size, step_size)

    if USE_AT_FOR_TESTING:
        X_test, y_test = create_dataset(df_test, window_size, step_size)
        X_train, y_train = X_train_full, y_train_full
    else:
        # Train/Test Split auf AR-Datensatz
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train_full)
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

    print(f"    Samples - Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Skalierung
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Label Encoding (falls AT-Datensatz)
    if USE_AT_FOR_TESTING:
        le = LabelEncoder()
        le.fit(np.concatenate([y_train, y_test]))
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    # Klassifikatoren trainieren und evaluieren
    classifiers = get_classifiers()

    for name, clf in classifiers.items():
        start = time.time()

        if name == "Random Forest":
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

        elapsed = time.time() - start
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({
            'window_size': window_size,
            'reaction_time_s': reaction_time,
            'classifier': name,
            'accuracy': acc,
            'f1_score': f1,
            'train_time': elapsed,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

        print(f"    {name}: {acc:.4f} ({100 * acc:.1f}%)")

# --- Ergebnisse speichern und visualisieren ---
print("\n[3/3] Ergebnisse...")

results_df = pd.DataFrame(results)
results_df.to_csv('window_size_results.csv', index=False)
print("      Ergebnisse gespeichert: window_size_results.csv")

# Zusammenfassungstabelle
print("\n" + "=" * 70)
print("ZUSAMMENFASSUNG: Accuracy nach Fenstergröße")
print("=" * 70)

pivot = results_df.pivot(index='window_size', columns='classifier', values='accuracy')
pivot['Reaktionszeit'] = pivot.index / SAMPLE_RATE_HZ
pivot = pivot[['Reaktionszeit', 'k-NN', 'SVM', 'Random Forest', 'MLP']]
print(pivot.round(4).to_string())

# --- Plot erstellen ---
plt.figure(figsize=(10, 6))

for classifier in ['k-NN', 'SVM', 'Random Forest', 'MLP']:
    data = results_df[results_df['classifier'] == classifier]
    plt.plot(data['reaction_time_s'], data['accuracy'], marker='o', label=classifier, linewidth=2)

plt.xlabel('Reaktionszeit (Sekunden)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Klassifikationsgenauigkeit vs. Reaktionszeit', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('window_size_comparison.png', dpi=150)
print("\n      Plot gespeichert: window_size_comparison.png")

# --- Fazit ---
print("\n" + "=" * 70)
print("FAZIT")
print("=" * 70)

best_per_window = results_df.loc[results_df.groupby('window_size')['accuracy'].idxmax()]
worst_window = results_df.groupby('window_size')['accuracy'].mean().idxmin()
worst_reaction = worst_window / SAMPLE_RATE_HZ

print(f"\nKleinste getestete Reaktionszeit: {WINDOW_SIZES[0] / SAMPLE_RATE_HZ:.1f}s (Fenstergröße {WINDOW_SIZES[0]})")
print(f"Größte getestete Reaktionszeit:  {WINDOW_SIZES[-1] / SAMPLE_RATE_HZ:.1f}s (Fenstergröße {WINDOW_SIZES[-1]})")

min_acc = results_df[results_df['window_size'] == WINDOW_SIZES[0]].groupby('classifier')['accuracy'].mean()
max_acc = results_df[results_df['window_size'] == WINDOW_SIZES[-1]].groupby('classifier')['accuracy'].mean()

print(f"\nAccuracy-Verlust (kürzeste vs. längste Reaktionszeit):")
for clf in ['k-NN', 'SVM', 'Random Forest', 'MLP']:
    diff = max_acc[clf] - min_acc[clf]
    print(f"  {clf}: {100 * diff:+.1f} Prozentpunkte")