"""
API REST del Simulador Interactivo de Politicas de Trabajo (Prediccion de Burnout).

Proyecto de Machine Learning - UPAO - Ing. de Sistemas e Inteligencia Artificial
Modelo: LightGBM (clasificacion multiclase Low/Medium/High) + explicacion SHAP.

Ejecutar en local:   uvicorn api_burnout:app --reload
En produccion:       uvicorn api_burnout:app --host 0.0.0.0 --port $PORT
"""
import os
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Carga de artefactos (modelo + scaler + label encoder + orden de variables)
# ---------------------------------------------------------------------------
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(RUTA_BASE, "burnout_model.pkl"), "rb") as f:
    artefactos = pickle.load(f)

modelo        = artefactos["model"]
scaler        = artefactos["scaler"]
label_encoder = artefactos["label_encoder"]
feature_cols  = artefactos["feature_cols"]

# SHAP es opcional: si no esta instalado, la API sigue funcionando sin explicacion
try:
    import shap
    explainer = shap.TreeExplainer(modelo)
    HAY_SHAP = True
except Exception:
    explainer = None
    HAY_SHAP = False

# Severidad por clase para el Burnout Risk Score (BRS 0-100)
SEVERIDAD = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
PESO_ORDEN = np.array([SEVERIDAD[c] for c in label_encoder.classes_])

app = FastAPI(
    title="API Simulador de Burnout",
    description="Prediccion del nivel de burnout de desarrolladores con LightGBM + SHAP",
    version="1.0",
)

# CORS abierto para que el frontend (mismo dominio u otro) pueda consumir la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PerfilDesarrollador(BaseModel):
    age: float
    experience_years: float
    daily_work_hours: float
    sleep_hours: float
    caffeine_intake: float
    bugs_per_day: float
    commits_per_day: float
    meetings_per_day: float
    screen_time: float
    exercise_hours: float
    stress_level: float


def _construir_features(perfil: dict) -> pd.DataFrame:
    """Aplica la MISMA ingenieria de caracteristicas usada en el entrenamiento."""
    d = dict(perfil)
    d["work_life_ratio"]    = d["daily_work_hours"] / (d["sleep_hours"] + 1e-9)
    d["productivity_score"] = d["commits_per_day"] * 0.6 + d["bugs_per_day"] * 0.4
    d["meeting_fatigue"]    = d["meetings_per_day"] * d["daily_work_hours"]
    d["recovery_index"]     = d["sleep_hours"] + d["exercise_hours"]
    return pd.DataFrame([[d[c] for c in feature_cols]], columns=feature_cols)


@app.post("/predict")
def predecir(perfil: PerfilDesarrollador):
    fila = _construir_features(perfil.dict())
    # transform (NO fit_transform): usa la media/std aprendidas en el entrenamiento
    fila_norm = pd.DataFrame(scaler.transform(fila), columns=feature_cols)

    clase_idx = int(modelo.predict(fila_norm)[0])
    proba     = modelo.predict_proba(fila_norm)[0]
    clase     = label_encoder.inverse_transform([clase_idx])[0]

    # Burnout Risk Score continuo (0 = sano, 100 = critico)
    brs = float(np.dot(proba, PESO_ORDEN) * 100)

    respuesta = {
        "burnout_predicho": clase,
        "burnout_risk_score": round(brs, 1),
        "probabilidades": {
            c: round(float(p) * 100, 1)
            for c, p in zip(label_encoder.classes_, proba)
        },
    }

    if HAY_SHAP:
        shap_vals = explainer.shap_values(fila_norm)
        contrib = dict(zip(feature_cols, shap_vals[0, :, clase_idx].tolist()))
        top3 = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        respuesta["explicacion_shap_top3"] = [
            {"variable": v, "contribucion": round(c, 4)} for v, c in top3
        ]

    return respuesta


@app.get("/health")
def salud():
    return {"status": "ok", "shap_disponible": HAY_SHAP}


@app.get("/")
def inicio():
    """Sirve el frontend del simulador What-If."""
    return FileResponse(os.path.join(RUTA_BASE, "static", "index.html"))
