

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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

"""
Erster Testlauf für Klasssifkatorenvergleiche auf dem WISDM-Datensatz.


"""

print("=" * 60)
print("WISDM Dataset - Klassifikatorvergleich")
print("=" * 60)

# --- Daten laden ---
print("\n[1/4] Lade Rohdaten...")

# WISDM Rohdaten laden (Format: user,activity,timestamp,x,y,z;)
data = []
with open("WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt", "r") as f:
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

df = pd.DataFrame(
    data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
print(f"Datenpunkte: {len(df):,}")

# Aktivitäten anzeigen
print(f"\n  Aktivitäten:")
for act, count in df['activity'].value_counts().items():
    print(f"    {act}: {count:,} ({100*count/len(df):.1f}%)")

# Feaures extrahierem
print("\n[2/4] Extrahiere Features aus Zeitfenstern")


def extract_features(window):
    features = []
    for axis in ['x', 'y', 'z']:
        values = window[axis].values
        features.extend([
            np.mean(values),           # Mittelwert
            np.std(values),            # Standardabweichung
            np.min(values),            # Minimum
            np.max(values),            # Maximum
            np.max(values) - np.min(values),  # Range
            np.median(values),         # Median
            stats.skew(values),        # Schiefe
            stats.kurtosis(values),    # Kurtosis
            np.sqrt(np.mean(values**2)),  # RMS
            np.sum(np.abs(np.diff(values)))  # Absolute Differenzensumme
        ])

    magnitude = np.sqrt(window['x']**2 + window['y']**2 + window['z']**2)
    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude) - np.min(magnitude)
    ])

    return features


# Zeitfenster: 200 Samples mit 50% Overlap (wie im Original-Paper)
window_size = 200
step_size = 100

X_list = []
y_list = []
user_list = []

# Gruppiere nach User und Aktivität für saubere Fenster
for (user, activity), group in df.groupby(['user', 'activity']):
    group = group.sort_values('timestamp').reset_index(drop=True)

    for start in range(0, len(group) - window_size + 1, step_size):
        window = group.iloc[start:start + window_size]
        features = extract_features(window)
        X_list.append(features)
        y_list.append(activity)
        user_list.append(user)

X = np.array(X_list)
y = np.array(y_list)
users = np.array(user_list)

# NaN/Inf Werte behandeln, sonst wirft einer der Klassifikatoren Fehler 
mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)) #Hack um NaN/Inf Werte zu entfernen
X = X[mask]
y = y[mask]
users = users[mask]
print(f"  Nach NaN-Bereinigung: {len(X):,} Samples")

# Feature-Namen
feature_names = []
for axis in ['x', 'y', 'z']:
    for feat in ['mean', 'std', 'min', 'max', 'range', 'median', 'skew', 'kurtosis', 'rms', 'abs_diff']:
        feature_names.append(f"{axis}_{feat}")
feature_names.extend(['mag_mean', 'mag_std', 'mag_range'])

print(f"  Extrahierte Samples: {len(X):,}")
print(f"  Features pro Sample: {X.shape[1]}")

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
activity_labels = {i: name for i, name in enumerate(le.classes_)}

print(f"\n  Klassenverteilung nach Feature-Extraktion:")
for label, name in activity_labels.items():
    count = np.sum(y_encoded == label)
    print(f"    {name}: {count} ({100*count/len(y_encoded):.1f}%)")

# --- Train/Test Split ---
print("\n[3/4] Erstelle Train/Test Split...")

#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"  Training: {len(X_train):,} Samples")
print(f"  Test:     {len(X_test):,} Samples")

# Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Klassifikatoren ---
print("\n[4/4] Training und Evaluation...")
print("=" * 60)

classifiers = {
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
}

results = []

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")

    # Training
    start_train = time.time()
    if name == "Random Forest":
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start_train

    # Vorhersage
    start_pred = time.time()
    if name == "Random Forest":
        y_pred = clf.predict(X_test)
    else:
        y_pred = clf.predict(X_test_scaled)
    pred_time = time.time() - start_pred

    # Metriken
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  Trainingszeit:   {train_time:.2f}s")
    print(f"  Vorhersagezeit:  {pred_time:.3f}s")
    print(f"  Accuracy:        {accuracy:.4f} ({100*accuracy:.2f}%)")

    results.append({
        'Klassifikator': name,
        'Accuracy': accuracy,
        'Trainingszeit (s)': train_time,
        'Vorhersagezeit (s)': pred_time
    })

# --- Zusammenfassung ---
print("\n" + "=" * 60)
print("Zusammenfassung")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
print("\n" + results_df.to_string(index=False))

best = results_df.iloc[0]
print(
    f"\n→ Bester Klassifikator: {best['Klassifikator']} mit {100*best['Accuracy']:.2f}% Accuracy")

# --- Detaillierter Report ---
print("\n" + "=" * 60)
print(f"Detaillierter Report: {best['Klassifikator']}")
print("=" * 60)

# Besten nochmal trainieren
if "Random Forest" in best['Klassifikator']:
    best_clf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
    best_clf.fit(X_train, y_train)
    y_pred_best = best_clf.predict(X_test)
elif "SVM" in best['Klassifikator']:
    best_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    best_clf.fit(X_train_scaled, y_train)
    y_pred_best = best_clf.predict(X_test_scaled)
elif "MLP" in best['Klassifikator']:
    best_clf = MLPClassifier(hidden_layer_sizes=(
        128, 64), max_iter=500, random_state=42)
    best_clf.fit(X_train_scaled, y_train)
    y_pred_best = best_clf.predict(X_test_scaled)
else:
    best_clf = KNeighborsClassifier(n_neighbors=5)
    best_clf.fit(X_train_scaled, y_train)
    y_pred_best = best_clf.predict(X_test_scaled)

print("\n Klassifkatpr-Report:")
target_names = [activity_labels[i] for i in sorted(activity_labels.keys())]
print(classification_report(y_test, y_pred_best, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
print(cm_df)

# Feature Importance (falls Random Forest)

print("\n" + "=" * 60)
print("Top 10 wichtigste Features (Random Forest)")
print("=" * 60)
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
