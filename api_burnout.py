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
from fastapi import FastAPI, Query
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

DERIVADAS = ["work_life_ratio", "productivity_score", "meeting_fatigue", "recovery_index"]

# Palancas del optimizador prescriptivo (mismas que el notebook):
#   nombre -> (variable, niveles graduales hacia una meta saludable, costo por nivel)
PALANCAS = {
    "Gestion del estres (programa de bienestar)": ("stress_level",     [80, 70, 60, 50, 40, 30], 2),
    "Jornada saludable (reducir horas)":          ("daily_work_hours", [10, 9, 8],               2),
    "Reducir reuniones (no-meeting days)":        ("meetings_per_day", [6, 4, 2],                1),
    "Higiene del sueno":                          ("sleep_hours",      [5.5, 6.5, 7.5, 8.0],     1),
    "Programa de ejercicio":                      ("exercise_hours",   [0.5, 1.0, 1.5],          1),
}

app = FastAPI(
    title="API Simulador de Burnout",
    description="Prediccion del nivel de burnout de desarrolladores con LightGBM + SHAP",
    version="2.0",
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


def _predecir_interno(perfil: dict):
    """Prediccion nucleo: devuelve (clase_idx, clase_texto, proba, brs, fila_norm)."""
    fila = _construir_features(perfil)
    # transform (NO fit_transform): usa la media/std aprendidas en el entrenamiento
    fila_norm = pd.DataFrame(scaler.transform(fila), columns=feature_cols)
    clase_idx = int(modelo.predict(fila_norm)[0])
    proba     = modelo.predict_proba(fila_norm)[0]
    clase     = label_encoder.inverse_transform([clase_idx])[0]
    brs       = float(np.dot(proba, PESO_ORDEN) * 100)
    return clase_idx, clase, proba, brs, fila_norm


def _brs_de_perfil(perfil: dict) -> float:
    """Solo el Burnout Risk Score de un perfil (para el optimizador)."""
    return _predecir_interno(perfil)[3]


@app.post("/predict")
def predecir(perfil: PerfilDesarrollador):
    datos = perfil.dict()
    clase_idx, clase, proba, brs, fila_norm = _predecir_interno(datos)

    # Valores de las 4 variables derivadas (para mostrarlas en el frontend)
    fila_orig = _construir_features(datos)
    derivadas = {v: round(float(fila_orig[v].iloc[0]), 2) for v in DERIVADAS}

    respuesta = {
        "burnout_predicho": clase,
        "burnout_risk_score": round(brs, 1),
        "probabilidades": {
            c: round(float(p) * 100, 1)
            for c, p in zip(label_encoder.classes_, proba)
        },
        "features_derivadas": derivadas,
    }

    if HAY_SHAP:
        shap_vals = explainer.shap_values(fila_norm)
        contrib = dict(zip(feature_cols, shap_vals[0, :, clase_idx].tolist()))
        orden = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        respuesta["explicacion_shap"] = [
            {"variable": v, "contribucion": round(c, 4)} for v, c in orden
        ]
        respuesta["explicacion_shap_top3"] = respuesta["explicacion_shap"][:3]

    return respuesta


@app.post("/optimize")
def optimizar(
    perfil: PerfilDesarrollador,
    objetivo_brs: float = Query(33, ge=0, le=100),
    costo_maximo: int = Query(25, ge=1, le=100),
):
    """
    Optimizador prescriptivo de politicas de bienestar (algoritmo voraz).
    En cada paso aplica la palanca con mayor potencial de reduccion del BRS,
    hasta alcanzar el objetivo o agotar el presupuesto de costo organizacional.
    """
    actual = perfil.dict()
    nivel  = {k: 0 for k in PALANCAS}
    plan, costo = [], 0

    r = _brs_de_perfil(actual)
    r_inicial = r

    while r > objetivo_brs and costo < costo_maximo:
        mejor = None  # (potencial_de_reduccion, nombre, var, niveles, costo)
        for nombre, (var, niveles, c) in PALANCAS.items():
            if nivel[nombre] >= len(niveles):
                continue
            # Potencial: cuanto bajaria el riesgo si esta palanca llegara a su meta final
            cand = dict(actual)
            cand[var] = niveles[-1]
            potencial = r - _brs_de_perfil(cand)
            if mejor is None or potencial > mejor[0]:
                mejor = (potencial, nombre, var, niveles, c)
        if mejor is None or mejor[0] <= 0.1:
            break
        _, nombre, var, niveles, c = mejor
        actual[var] = niveles[nivel[nombre]]
        nr = _brs_de_perfil(actual)
        plan.append({
            "accion": nombre,
            "variable": var,
            "nuevo_valor": actual[var],
            "brs_antes": round(r, 1),
            "brs_despues": round(nr, 1),
            "costo": c,
        })
        costo += c
        nivel[nombre] += 1
        r = nr

    original = perfil.dict()
    cambios = {
        k: {"antes": original[k], "despues": actual[k]}
        for k in original if original[k] != actual[k]
    }

    return {
        "brs_inicial": round(r_inicial, 1),
        "brs_final": round(r, 1),
        "reduccion": round(r_inicial - r, 1),
        "costo_total": costo,
        "objetivo_brs": objetivo_brs,
        "objetivo_alcanzado": r <= objetivo_brs,
        "plan": plan,
        "cambios_netos": cambios,
    }


@app.get("/health")
def salud():
    return {"status": "ok", "shap_disponible": HAY_SHAP}


@app.get("/")
def inicio():
    """Sirve el frontend del simulador What-If."""
    return FileResponse(os.path.join(RUTA_BASE, "static", "index.html"))
