"""
Enriquece burnout_model.pkl con una RED NEURONAL (MLP) real como segundo modelo
y una AUDITORIA DE EQUIDAD por subgrupos demograficos, sin alterar el LightGBM ya
desplegado (se reutiliza tal cual del pkl existente).

Reproduce el pipeline EXACTO del notebook (mismas features, mismo scaler, misma
particion con random_state=42) verificando que el LightGBM del pkl da 0.9819 en test.

Ejecutar:  python entrenar_red_neuronal.py
"""
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42
RUTA_PKL = "burnout_model.pkl"
RUTA_DATASET = "df_burnout_procesado.csv"

# ---------------------------------------------------------------------------
# 1. Reproducir el pipeline exacto del notebook
# ---------------------------------------------------------------------------
with open(RUTA_PKL, "rb") as f:
    art = pickle.load(f)

modelo        = art["model"]           # LightGBM ya entrenado y desplegado (NO se toca)
scaler        = art["scaler"]
label_encoder = art["label_encoder"]
feature_cols  = art["feature_cols"]

df = pd.read_csv(RUTA_DATASET)
d = df.copy()
d["work_life_ratio"]    = d["daily_work_hours"] / (d["sleep_hours"] + 1e-9)
d["productivity_score"] = d["commits_per_day"] * 0.6 + d["bugs_per_day"] * 0.4
d["meeting_fatigue"]    = d["meetings_per_day"] * d["daily_work_hours"]
d["recovery_index"]     = d["sleep_hours"] + d["exercise_hours"]

y = label_encoder.transform(d["burnout_level"])
X = d[feature_cols].copy()
X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp)

acc_lgbm = accuracy_score(y_test, modelo.predict(X_test))
assert abs(acc_lgbm - 0.9819) < 0.002, f"Reproduccion incorrecta: acc={acc_lgbm}"
print(f"[OK] Pipeline reproducido. LightGBM en test: acc={acc_lgbm:.4f}")

# ---------------------------------------------------------------------------
# 2. Entrenar la RED NEURONAL (MLP real: 3 capas ocultas, ReLU, backpropagation)
# ---------------------------------------------------------------------------
nn = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
    alpha=1e-4, learning_rate_init=0.001, max_iter=300,
    early_stopping=True, random_state=RANDOM_STATE,
)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
acc_nn = accuracy_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn, average="macro")
print(f"[OK] Red neuronal (MLP) entrenada: acc={acc_nn:.4f} f1={f1_nn:.4f} | "
      f"capas={nn.hidden_layer_sizes} | iteraciones={nn.n_iter_}")

# Acuerdo entre los dos modelos sobre el test
y_pred_lgbm = modelo.predict(X_test)
acuerdo = float(np.mean(y_pred_lgbm == y_pred_nn))
print(f"[OK] Acuerdo LightGBM <-> Red neuronal en test: {acuerdo*100:.2f}%")

# ---------------------------------------------------------------------------
# 3. Auditoria de EQUIDAD por subgrupos (mitigacion/deteccion de sesgos)
#    Usamos el indice original para recuperar age/experience_years sin escalar.
# ---------------------------------------------------------------------------
def metricas_subgrupo(mask):
    idx = X_test.index[mask]
    yt = y_test[mask]
    yp = modelo.predict(X_test.loc[idx])
    return {"n": int(mask.sum()),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "f1_macro": round(float(f1_score(yt, yp, average="macro")), 4)}

age_test = df.loc[X_test.index, "age"].values
exp_test = df.loc[X_test.index, "experience_years"].values

fairness = {
    "edad_20_27":       metricas_subgrupo((age_test >= 20) & (age_test <= 27)),
    "edad_28_36":       metricas_subgrupo((age_test >= 28) & (age_test <= 36)),
    "edad_37_44":       metricas_subgrupo((age_test >= 37) & (age_test <= 44)),
    "experiencia_0_4":  metricas_subgrupo((exp_test >= 0) & (exp_test <= 4)),
    "experiencia_5_11": metricas_subgrupo((exp_test >= 5) & (exp_test <= 11)),
    "experiencia_12_19":metricas_subgrupo((exp_test >= 12) & (exp_test <= 19)),
}
f1s = [v["f1_macro"] for v in fairness.values()]
brecha = round(max(f1s) - min(f1s), 4)
print(f"[OK] Auditoria de equidad. Brecha maxima de F1 entre subgrupos: {brecha}")
for k, v in fairness.items():
    print(f"     {k:20s} n={v['n']:4d}  acc={v['accuracy']}  f1={v['f1_macro']}")

# ---------------------------------------------------------------------------
# 4. Re-exportar el pkl enriquecido (LightGBM intacto + red neuronal + auditorias)
# ---------------------------------------------------------------------------
art_nuevo = dict(art)
art_nuevo["nn_model"] = nn
art_nuevo["metricas_modelos"] = {
    "lightgbm": {"accuracy": round(float(acc_lgbm), 4),
                 "f1_macro": round(float(f1_score(y_test, y_pred_lgbm, average="macro")), 4)},
    "red_neuronal": {"accuracy": round(float(acc_nn), 4), "f1_macro": round(float(f1_nn), 4),
                     "arquitectura": "MLP 64-32-16 (ReLU, Adam, early stopping)"},
    "acuerdo": round(acuerdo, 4),
}
art_nuevo["auditoria_equidad"] = {"subgrupos": fairness, "brecha_max_f1": brecha}

with open(RUTA_PKL, "wb") as f:
    pickle.dump(art_nuevo, f)

import os
print(f"\n[OK] pkl re-exportado ({os.path.getsize(RUTA_PKL)/1024:.1f} KB) con claves: {list(art_nuevo.keys())}")
