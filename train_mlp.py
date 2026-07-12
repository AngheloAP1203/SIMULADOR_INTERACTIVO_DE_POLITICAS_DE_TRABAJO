"""
Entrena un Perceptron Multicapa (MLPClassifier) como segundo modelo -
esta vez una red neuronal real- para comparar contra el LightGBM ya
desplegado en burnout_model.pkl. Reutiliza el MISMO scaler, feature_cols
y label_encoder ya ajustados, para que ambos modelos sean directamente
comparables sobre la misma representacion de features.

Ejecutar una sola vez (o cuando cambie el dataset):
    python train_mlp.py
Genera burnout_mlp.pkl, consumido de forma opcional por api_burnout.py.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

RUTA_MODELO_BASE = "burnout_model.pkl"
RUTA_DATASET = "df_burnout_procesado.csv"
RUTA_SALIDA = "burnout_mlp.pkl"

with open(RUTA_MODELO_BASE, "rb") as f:
    artefactos = pickle.load(f)

scaler = artefactos["scaler"]
label_encoder = artefactos["label_encoder"]
feature_cols = artefactos["feature_cols"]

df = pd.read_csv(RUTA_DATASET)


def construir_features(df):
    d = df.copy()
    d["work_life_ratio"] = d["daily_work_hours"] / (d["sleep_hours"] + 1e-9)
    d["productivity_score"] = d["commits_per_day"] * 0.6 + d["bugs_per_day"] * 0.4
    d["meeting_fatigue"] = d["meetings_per_day"] * d["daily_work_hours"]
    d["recovery_index"] = d["sleep_hours"] + d["exercise_hours"]
    return d[feature_cols]


X = construir_features(df)
y = label_encoder.transform(df["burnout_level"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train_norm = pd.DataFrame(scaler.transform(X_train), columns=feature_cols)
X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=42,
)
mlp.fit(X_train_norm, y_train)

y_pred = mlp.predict(X_test_norm)
y_proba = mlp.predict_proba(X_test_norm)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

print(f"MLP - Accuracy: {acc:.4f}")
print(f"MLP - F1-macro: {f1m:.4f}")
print(f"MLP - AUC-ROC (ovr macro): {auc:.4f}")
print(f"MLP - Iteraciones hasta converger: {mlp.n_iter_}")
print(f"MLP - Arquitectura: {mlp.hidden_layer_sizes}")

with open(RUTA_SALIDA, "wb") as f:
    pickle.dump(
        {
            "model": mlp,
            "feature_cols": feature_cols,
            "metricas_test": {
                "accuracy": round(float(acc), 4),
                "f1_macro": round(float(f1m), 4),
                "auc_roc_ovr_macro": round(float(auc), 4),
                "n_test": int(len(y_test)),
                "arquitectura": list(mlp.hidden_layer_sizes),
            },
        },
        f,
    )

print(f"\nGuardado en {RUTA_SALIDA}")
