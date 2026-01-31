
"""
WISDM Dataset - Klassifikatorvergleich mit verschiedenen Fenstergrößen
Untersucht den Einfluss der Zeitfenstergröße auf die Klassifikationsgenauigkeit
"""

import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy import stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- Feature-Namen definieren ---
FEATURE_NAMES = []
for axis in ['x', 'y', 'z']:
    for feat in ['mean', 'std', 'min', 'max', 'range', 'median', 'skew', 'kurtosis', 'rms', 'abs_diff']:
        FEATURE_NAMES.append(f"{axis}_{feat}")
FEATURE_NAMES.extend(['mag_mean', 'mag_std', 'mag_range'])

# --- Daten laden ---
print("=" * 60)
print("WISDM Dataset - Fenstergrößen-Experiment")
print("=" * 60)

print("\n[1/4] Lade Rohdaten...")

data = []
with open("data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt", "r") as f:
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

df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
print(f"  Geladene Datenpunkte: {len(df):,}")

# Aktivitäten anzeigen
print(f"\n  Aktivitäten:")
for act, count in df['activity'].value_counts().items():
    print(f"    {act}: {count:,} ({100 * count / len(df):.1f}%)")


# --- Feature-Extraktion Funktion ---
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

    magnitude = np.sqrt(window['x'] ** 2 + window['y'] ** 2 + window['z'] ** 2)
    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude) - np.min(magnitude)
    ])

    return features


def create_dataset(df, window_size, step_size=None):
    """Erstellt Datensatz mit gegebener Fenstergröße"""
    if step_size is None:
        step_size = window_size // 2  # 50% Overlap

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


def evaluate_classifiers(X_train, X_test, y_train, y_test, compute_feature_importance=False):
    """Evaluiert alle Klassifikatoren und gibt Ergebnisse zurück"""

    # Skalierung
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
    }

    results = {}

    for name, clf in classifiers.items():
        start_train = time.time()
        if name == "Random Forest":
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        train_time = time.time() - start_train

        accuracy = accuracy_score(y_test, y_pred)

        # Feature Importance berechnen
        feature_importance = None
        if compute_feature_importance:
            if name == "Random Forest":
                # Random Forest hat eingebaute Feature Importance
                feature_importance = clf.feature_importances_
            else:
                # Für andere Klassifikatoren: Permutation Importance
                if name == "Random Forest":
                    perm_result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                else:
                    perm_result = permutation_importance(clf, X_test_scaled, y_test, n_repeats=10, random_state=42,
                                                         n_jobs=-1)
                feature_importance = perm_result.importances_mean

        results[name] = {
            'accuracy': accuracy,
            'train_time': train_time,
            'classifier': clf,
            'feature_importance': feature_importance,
            'y_pred': y_pred
        }

    return results, scaler


def print_feature_importance(results, top_n=10):
    """Gibt die Feature Importance für alle Klassifikatoren aus"""
    print("\n" + "=" * 60)
    print(f"Feature Importance (Top {top_n})")
    print("=" * 60)

    for name, metrics in results.items():
        if metrics['feature_importance'] is not None:
            print(f"\n--- {name} ---")
            importances = metrics['feature_importance']
            indices = np.argsort(importances)[::-1][:top_n]
            for i, idx in enumerate(indices):
                print(f"  {i + 1:2}. {FEATURE_NAMES[idx]:15} {importances[idx]:.4f}")


# --- Experiment: Verschiedene Fenstergrößen ---
print("\n[2/4] Starte Fenstergrößen-Experiment...")

window_sizes = [50, 100, 150, 200, 250, 300, 400]
all_results = []

for window_size in window_sizes:
    print(f"\n--- Fenstergröße: {window_size} ---")

    # Datensatz erstellen
    X, y = create_dataset(df, window_size)
    print(f"  Samples: {len(X):,}")

    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    activity_labels = {i: name for i, name in enumerate(le.classes_)}

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # Klassifikatoren evaluieren (Feature Importance nur für Fenstergröße 200)
    compute_fi = (window_size == 200)
    results, scaler = evaluate_classifiers(X_train, X_test, y_train, y_test, compute_feature_importance=compute_fi)

    for clf_name, metrics in results.items():
        all_results.append({
            'window_size': window_size,
            'classifier': clf_name,
            'accuracy': metrics['accuracy'],
            'train_time': metrics['train_time']
        })
        print(f"  {clf_name}: {100 * metrics['accuracy']:.2f}%")

    # Feature Importance ausgeben (nur für Fenstergröße 200)
    if compute_fi:
        print_feature_importance(results)

        # Detaillierten Report für besten Klassifikator ausgeben
        best_clf_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_metrics = results[best_clf_name]

        print("\n" + "=" * 60)
        print(f"Detaillierter Report: {best_clf_name} (Fenstergröße {window_size})")
        print("=" * 60)

        target_names = [activity_labels[i] for i in sorted(activity_labels.keys())]
        print("\nClassification Report:")
        print(classification_report(y_test, best_metrics['y_pred'], target_names=target_names))

        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, best_metrics['y_pred'])
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm_df)

# --- Ergebnisse zusammenfassen ---
print("\n" + "=" * 60)
print("[3/4] Ergebnisse Zusammenfassung")
print("=" * 60)

results_df = pd.DataFrame(all_results)

# Pivot-Tabelle für Übersicht
pivot_accuracy = results_df.pivot(index='window_size', columns='classifier', values='accuracy')
print("\nAccuracy nach Fenstergröße (%):")
print((pivot_accuracy * 100).round(2).to_string())

# Beste Konfiguration pro Klassifikator
print("\nBeste Fenstergröße pro Klassifikator:")
for clf_name in ['k-NN', 'SVM', 'Random Forest', 'MLP']:
    clf_data = results_df[results_df['classifier'] == clf_name]
    best_idx = clf_data['accuracy'].idxmax()
    best = clf_data.loc[best_idx]
    print(f"  {clf_name}: Fenstergröße {int(best['window_size'])} ({100 * best['accuracy']:.2f}%)")

# Beste Gesamtkonfiguration
best_idx = results_df['accuracy'].idxmax()
best = results_df.loc[best_idx]
print(f"\n→ Beste Gesamtkonfiguration: {best['classifier']} mit Fenstergröße {int(best['window_size'])}")
print(f"  Accuracy: {100 * best['accuracy']:.2f}%")

# --- Visualisierung ---
print("\n[4/4] Erstelle Visualisierungen...")

# Plot 1: Accuracy vs Fenstergröße
plt.figure(figsize=(10, 6))
colors = {'k-NN': '#1f77b4', 'SVM': '#ff7f0e', 'Random Forest': '#2ca02c', 'MLP': '#d62728'}

for clf_name in ['k-NN', 'SVM', 'Random Forest', 'MLP']:
    clf_data = results_df[results_df['classifier'] == clf_name]
    plt.plot(clf_data['window_size'], clf_data['accuracy'] * 100,
             marker='o', label=clf_name, color=colors[clf_name], linewidth=2, markersize=8)

plt.xlabel('Fenstergröße (Samples)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Klassifikationsgenauigkeit nach Fenstergröße', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.xticks(window_sizes)
plt.ylim(bottom=80)

plt.tight_layout()
plt.savefig('window_size_comparison.png', dpi=150)
print("  Plot gespeichert: window_size_comparison.png")

# Plot 2: Trainingszeit vs Fenstergröße
plt.figure(figsize=(10, 6))

for clf_name in ['k-NN', 'SVM', 'Random Forest', 'MLP']:
    clf_data = results_df[results_df['classifier'] == clf_name]
    plt.plot(clf_data['window_size'], clf_data['train_time'],
             marker='s', label=clf_name, color=colors[clf_name], linewidth=2, markersize=8)

plt.xlabel('Fenstergröße (Samples)', fontsize=12)
plt.ylabel('Trainingszeit (Sekunden)', fontsize=12)
plt.title('Trainingszeit nach Fenstergröße', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(window_sizes)

plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=150)
print("  Plot gespeichert: training_time_comparison.png")

# Ergebnisse als CSV speichern
results_df.to_csv('window_size_results.csv', index=False)
print("  Ergebnisse gespeichert: window_size_results.csv")

print("\n" + "=" * 60)
print("Fertig!")
print("=" * 60)