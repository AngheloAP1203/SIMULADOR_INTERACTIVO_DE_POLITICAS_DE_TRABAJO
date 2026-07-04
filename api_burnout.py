"""
API REST del Simulador Interactivo de Politicas de Trabajo (Prediccion de Burnout).

Proyecto de Machine Learning - UPAO - Ing. de Sistemas e Inteligencia Artificial
Modelo: LightGBM (clasificacion multiclase Low/Medium/High) + explicacion SHAP.

Ejecutar en local:   uvicorn api_burnout:app --reload
En produccion:       uvicorn api_burnout:app --host 0.0.0.0 --port $PORT
"""
import io
import os
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy.stats import percentileofscore

# ---------------------------------------------------------------------------
# Carga de artefactos (modelo + scaler + label encoder + orden de variables)
# ---------------------------------------------------------------------------
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(RUTA_BASE, "burnout_model.pkl")

with open(RUTA_MODELO, "rb") as f:
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

# Dataset REAL usado en el entrenamiento (7,000 registros). Se usa para calcular
# percentiles poblacionales y rangos reales del analisis de sensibilidad.
# Carga protegida: si el CSV no esta presente en el entorno de despliegue, esas
# dos funcionalidades se desactivan con gracia (el resto de la API sigue igual).
RUTA_DATASET = os.path.join(RUTA_BASE, "df_burnout_procesado.csv")
try:
    df_entrenamiento = pd.read_csv(RUTA_DATASET)
    HAY_DATASET = True
except Exception:
    df_entrenamiento = None
    HAY_DATASET = False

VARIABLES_ORIGINALES = [
    "age", "experience_years", "daily_work_hours", "sleep_hours", "caffeine_intake",
    "bugs_per_day", "commits_per_day", "meetings_per_day", "screen_time",
    "exercise_hours", "stress_level",
]

# Metricas REALES obtenidas en el notebook sobre el conjunto de prueba (15% held-out,
# 1,050 registros nunca vistos durante el entrenamiento). No se recalculan aqui.
METRICAS_TEST = {
    "accuracy": 0.9819,
    "f1_macro": 0.9821,
    "auc_roc_ovr_macro": 0.9953,
    "log_loss": 0.0686,
    "n_test": 1050,
}

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
    version="3.0",
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


def _percentiles(perfil: dict) -> dict | None:
    """
    Compara cada variable del perfil contra los 7,000 registros REALES usados en el
    entrenamiento (df_entrenamiento). Devuelve, por variable, el percentil real que
    ocupa el valor ingresado dentro de esa poblacion (0-100).
    """
    if not HAY_DATASET:
        return None
    resultado = {}
    for var in VARIABLES_ORIGINALES:
        pct = percentileofscore(df_entrenamiento[var], perfil[var], kind="mean")
        resultado[var] = round(float(pct), 1)
    return resultado


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
        "comparacion_poblacional": _percentiles(datos),
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


@app.post("/predict/batch")
async def predecir_lote(archivo: UploadFile = File(...)):
    """
    Prediccion por lotes: recibe un CSV con las 11 columnas de PerfilDesarrollador
    (una fila por desarrollador) y aplica el MISMO pipeline ya entrenado a cada fila.
    """
    contenido = await archivo.read()
    df_lote = pd.read_csv(io.BytesIO(contenido))

    faltantes = [c for c in VARIABLES_ORIGINALES if c not in df_lote.columns]
    if faltantes:
        return {"error": f"Faltan columnas requeridas: {faltantes}", "columnas_esperadas": VARIABLES_ORIGINALES}

    resultados = []
    for i, fila in df_lote.iterrows():
        perfil = {c: float(fila[c]) for c in VARIABLES_ORIGINALES}
        _, clase, proba, brs, _ = _predecir_interno(perfil)
        resultados.append({
            "fila": int(i) + 1,
            "burnout_predicho": clase,
            "burnout_risk_score": round(brs, 1),
            "probabilidades": {c: round(float(p) * 100, 1) for c, p in zip(label_encoder.classes_, proba)},
        })

    conteo = {c: sum(1 for r in resultados if r["burnout_predicho"] == c) for c in label_encoder.classes_}
    brs_promedio = round(float(np.mean([r["burnout_risk_score"] for r in resultados])), 1)

    return {
        "total_registros": len(resultados),
        "resumen": {
            "conteo_por_clase": conteo,
            "brs_promedio_equipo": brs_promedio,
            "en_riesgo_alto": sum(1 for r in resultados if r["burnout_risk_score"] >= 66),
        },
        "resultados": resultados,
    }


@app.get("/model/info")
def info_modelo():
    """Ficha tecnica real del modelo ya entrenado (parametros y metricas reales)."""
    return {
        "algoritmo": type(modelo).__name__,
        "clases": list(label_encoder.classes_),
        "n_variables": len(feature_cols),
        "variables": feature_cols,
        "arboles_usados": int(getattr(modelo, "best_iteration_", modelo.get_params().get("n_estimators"))),
        "hiperparametros": {
            k: v for k, v in modelo.get_params().items()
            if k in ("n_estimators", "learning_rate", "max_depth", "num_leaves",
                      "min_child_samples", "subsample", "colsample_bytree",
                      "reg_alpha", "reg_lambda", "class_weight", "objective")
        },
        "tamano_pkl_kb": round(os.path.getsize(RUTA_MODELO) / 1024, 1),
        "metricas_test": METRICAS_TEST,
        "shap_disponible": HAY_SHAP,
        "dataset_referencia_disponible": HAY_DATASET,
    }


@app.get("/model/importance")
def importancia_modelo():
    """Importancia de variables NATIVA de LightGBM (feature_importances_, ya calculada en el entrenamiento)."""
    pares = sorted(
        zip(feature_cols, modelo.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    )
    return {"tipo": "split (numero de veces que la variable se usa para dividir un arbol)",
            "importancias": [{"variable": v, "importancia": int(i)} for v, i in pares]}


@app.post("/sensitivity")
def analisis_sensibilidad(
    perfil: PerfilDesarrollador,
    variable: str = Query(..., description="Variable a barrer, una de: " + ", ".join(VARIABLES_ORIGINALES)),
    n_puntos: int = Query(20, ge=5, le=50),
):
    """
    Curva de dependencia parcial simplificada: recorre 'variable' entre su minimo y
    maximo REAL (segun el dataset de entrenamiento) manteniendo el resto del perfil
    fijo, y llama al modelo YA entrenado en cada punto.
    """
    if variable not in VARIABLES_ORIGINALES:
        return {"error": f"Variable invalida. Debe ser una de: {VARIABLES_ORIGINALES}"}

    if HAY_DATASET:
        v_min = float(df_entrenamiento[variable].min())
        v_max = float(df_entrenamiento[variable].max())
    else:
        # Respaldo: usa el rango +/-50% del valor actual si no hay dataset de referencia
        actual = perfil.dict()[variable]
        v_min, v_max = actual * 0.5, actual * 1.5

    base = perfil.dict()
    puntos = np.linspace(v_min, v_max, n_puntos)
    curva = []
    for valor in puntos:
        cand = dict(base)
        cand[variable] = float(valor)
        _, clase, _, brs, _ = _predecir_interno(cand)
        curva.append({"valor": round(float(valor), 2), "burnout_risk_score": round(brs, 1), "clase_predicha": clase})

    return {"variable": variable, "rango_real_dataset": [round(v_min, 2), round(v_max, 2)], "curva": curva}


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
    return {"status": "ok", "shap_disponible": HAY_SHAP, "dataset_referencia_disponible": HAY_DATASET}


@app.get("/")
def inicio():
    """Sirve el frontend del simulador What-If."""
    return FileResponse(os.path.join(RUTA_BASE, "static", "index.html"))
