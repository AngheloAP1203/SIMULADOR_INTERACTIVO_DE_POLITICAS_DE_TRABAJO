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
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from scipy.stats import percentileofscore
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler, label_binarize

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

# Red neuronal de SEGUNDA OPINION (MLP 64-32-16 entrenada sobre los mismos datos, incluida
# dentro de burnout_model.pkl). Dos familias de modelos distintas (boosting de arboles vs
# red neuronal): si coinciden, la prediccion es mas confiable; si discrepan, se marca para
# revision humana. Opcional: si el pkl no la trae, la API funciona solo con LightGBM.
red_neuronal      = artefactos.get("nn_model")
METRICAS_MODELOS  = artefactos.get("metricas_modelos", {})
AUDITORIA_EQUIDAD = artefactos.get("auditoria_equidad")
NN_INFO           = METRICAS_MODELOS.get("red_neuronal", {}) if METRICAS_MODELOS else {}
HAY_NN            = red_neuronal is not None

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

# ---------------------------------------------------------------------------
# Segmentacion no supervisada (K-Means), misma metodologia y variables del
# notebook (Seccion 5.4), reentrenada aqui sobre el dataset real de referencia.
# ---------------------------------------------------------------------------
VARS_SEGMENTACION = ["daily_work_hours", "sleep_hours", "stress_level",
                     "meetings_per_day", "exercise_hours", "screen_time"]
try:
    if not HAY_DATASET:
        raise RuntimeError("dataset de referencia no disponible")
    scaler_seg = StandardScaler().fit(df_entrenamiento[VARS_SEGMENTACION])
    X_seg = scaler_seg.transform(df_entrenamiento[VARS_SEGMENTACION])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_seg)

    # Etiquetas interpretables ordenadas por estres promedio (igual que el notebook)
    perfil_seg = (
        df_entrenamiento.assign(_seg=kmeans.labels_)
        .groupby("_seg")[VARS_SEGMENTACION].mean()
    )
    orden_riesgo = perfil_seg["stress_level"].sort_values()
    ETIQUETAS_SEGMENTO = ["Equilibrado / bajo riesgo", "Carga moderada", "Alta carga", "Sobrecargado / alto riesgo"]
    NOMBRES_SEGMENTO = {seg: etq for etq, seg in zip(ETIQUETAS_SEGMENTO, orden_riesgo.index)}
    HAY_KMEANS = True
except Exception:
    scaler_seg, kmeans, NOMBRES_SEGMENTO = None, None, {}
    HAY_KMEANS = False

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


def _construir_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Version vectorizada de _construir_features para procesar varias filas a la vez."""
    d = df.copy()
    d["work_life_ratio"]    = d["daily_work_hours"] / (d["sleep_hours"] + 1e-9)
    d["productivity_score"] = d["commits_per_day"] * 0.6 + d["bugs_per_day"] * 0.4
    d["meeting_fatigue"]    = d["meetings_per_day"] * d["daily_work_hours"]
    d["recovery_index"]     = d["sleep_hours"] + d["exercise_hours"]
    return d[feature_cols]


def _segmento_de_perfil(perfil: dict) -> dict | None:
    """
    Asigna el perfil a uno de los 4 arquetipos de comportamiento laboral hallados
    por K-Means (aprendizaje NO supervisado), complementario a la clasificacion
    supervisada de LightGBM.
    """
    if not HAY_KMEANS:
        return None
    fila = pd.DataFrame([[perfil[v] for v in VARS_SEGMENTACION]], columns=VARS_SEGMENTACION)
    fila_esc = scaler_seg.transform(fila)
    seg_id = int(kmeans.predict(fila_esc)[0])
    return {"segmento_id": seg_id, "segmento_nombre": NOMBRES_SEGMENTO[seg_id]}


NOMBRES_LEGIBLES = {
    "age": "la edad", "experience_years": "los años de experiencia",
    "daily_work_hours": "las horas de trabajo diario", "sleep_hours": "las horas de sueño",
    "caffeine_intake": "el consumo de cafeína", "bugs_per_day": "los bugs resueltos por día",
    "commits_per_day": "los commits por día", "meetings_per_day": "las reuniones diarias",
    "screen_time": "el tiempo frente a pantalla", "exercise_hours": "las horas de ejercicio",
    "stress_level": "el nivel de estrés", "work_life_ratio": "el desequilibrio trabajo/descanso",
    "productivity_score": "la carga de trabajo técnico", "meeting_fatigue": "la fatiga por reuniones",
    "recovery_index": "la capacidad de recuperación",
}
NIVEL_LEGIBLE = {"Low": "BAJO", "Medium": "MEDIO", "High": "ALTO"}


def _explicacion_natural(clase, brs, shap_items, segunda_opinion):
    """
    Servicio cognitivo de explicacion: convierte la salida numerica del modelo (SHAP)
    en una explicacion en lenguaje natural, generada deterministicamente a partir de
    las contribuciones reales de cada variable (sin inventar nada).
    """
    frases = [f"El modelo clasifica este perfil con riesgo de burnout {NIVEL_LEGIBLE.get(clase, clase)} "
              f"(índice {brs:.0f} de 100)."]
    if shap_items:
        suben = [x for x in shap_items if x["contribucion"] > 0][:3]
        bajan = [x for x in shap_items if x["contribucion"] < 0][:2]
        if suben:
            frases.append("Los factores que más elevan este riesgo son "
                          + ", ".join(NOMBRES_LEGIBLES.get(x["variable"], x["variable"]) for x in suben) + ".")
        if bajan:
            frases.append("En sentido contrario, "
                          + " y ".join(NOMBRES_LEGIBLES.get(x["variable"], x["variable"]) for x in bajan)
                          + " reducen la estimación.")
    if segunda_opinion is not None:
        if segunda_opinion["coincide"]:
            frases.append("La red neuronal de verificación coincide con esta clasificación, "
                          "lo que refuerza la confiabilidad del resultado.")
        else:
            frases.append(f"Atención: la red neuronal de verificación clasifica este perfil como "
                          f"{NIVEL_LEGIBLE.get(segunda_opinion['burnout_predicho'], '?')} — al haber discrepancia "
                          f"entre los dos modelos, se recomienda revisión humana de este caso.")
    return " ".join(frases)


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
        "segmento": _segmento_de_perfil(datos),
    }

    # Segunda opinion: red neuronal (familia de modelo distinta) sobre el mismo perfil
    segunda = None
    if HAY_NN:
        clase_nn = label_encoder.inverse_transform([int(red_neuronal.predict(fila_norm)[0])])[0]
        segunda = {
            "modelo": "Red Neuronal MLP (64-32-16)",
            "burnout_predicho": clase_nn,
            "coincide": bool(clase_nn == clase),
        }
        respuesta["segunda_opinion_red_neuronal"] = segunda

    shap_items = None
    if HAY_SHAP:
        shap_vals = explainer.shap_values(fila_norm)
        contrib = dict(zip(feature_cols, shap_vals[0, :, clase_idx].tolist()))
        orden = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        shap_items = [{"variable": v, "contribucion": round(c, 4)} for v, c in orden]
        respuesta["explicacion_shap"] = shap_items
        respuesta["explicacion_shap_top3"] = shap_items[:3]

    # Servicio cognitivo de explicacion en lenguaje natural (generado desde SHAP real)
    respuesta["explicacion_texto"] = _explicacion_natural(clase, brs, shap_items, segunda)

    return respuesta


@app.post("/predict/batch")
async def predecir_lote(archivo: UploadFile = File(...)):
    """
    Prediccion por lotes: recibe un CSV con las 11 columnas de PerfilDesarrollador
    (una fila por desarrollador) y aplica el MISMO pipeline ya entrenado a todas las
    filas de una vez (vectorizado), incluyendo la clasificacion supervisada (LightGBM)
    y el segmento no supervisado (K-Means) de cada persona.
    """
    contenido = await archivo.read()
    df_lote = pd.read_csv(io.BytesIO(contenido))

    faltantes = [c for c in VARIABLES_ORIGINALES if c not in df_lote.columns]
    if faltantes:
        return {"error": f"Faltan columnas requeridas: {faltantes}", "columnas_esperadas": VARIABLES_ORIGINALES}

    df_feat = _construir_features_batch(df_lote[VARIABLES_ORIGINALES])
    X_norm = pd.DataFrame(scaler.transform(df_feat), columns=feature_cols)
    y_pred = modelo.predict(X_norm)
    y_proba = modelo.predict_proba(X_norm)
    brs = y_proba @ PESO_ORDEN * 100

    if HAY_KMEANS:
        X_seg = scaler_seg.transform(df_lote[VARS_SEGMENTACION])
        segmentos = kmeans.predict(X_seg)
    else:
        segmentos = [None] * len(df_lote)

    resultados = []
    for i in range(len(df_lote)):
        clase = label_encoder.inverse_transform([int(y_pred[i])])[0]
        fila_result = {
            "fila": i + 1,
            "burnout_predicho": clase,
            "burnout_risk_score": round(float(brs[i]), 1),
            "probabilidades": {c: round(float(p) * 100, 1) for c, p in zip(label_encoder.classes_, y_proba[i])},
        }
        if HAY_KMEANS:
            fila_result["segmento"] = NOMBRES_SEGMENTO[int(segmentos[i])]
        resultados.append(fila_result)

    conteo = {c: int(sum(1 for r in resultados if r["burnout_predicho"] == c)) for c in label_encoder.classes_}
    brs_promedio = round(float(np.mean(brs)), 1)

    return {
        "total_registros": len(resultados),
        "resumen": {
            "conteo_por_clase": conteo,
            "brs_promedio_equipo": brs_promedio,
            "en_riesgo_alto": int(sum(1 for r in resultados if r["burnout_risk_score"] >= 66)),
        },
        "resultados": resultados,
    }


@app.post("/evaluate")
async def evaluar_modelo(archivo: UploadFile = File(...)):
    """
    Evaluacion supervisada del modelo YA entrenado contra datos reales etiquetados.
    El CSV debe incluir las 11 columnas del perfil MAS la columna 'burnout_level'
    con la clase real (Low/Medium/High). Calcula, con sklearn, las mismas metricas
    usadas en el notebook: accuracy, F1-macro, AUC-ROC, matriz de confusion, reporte
    por clase y curvas ROC (todo sobre datos que el modelo nunca vio en el entrenamiento).
    """
    contenido = await archivo.read()
    df_eval = pd.read_csv(io.BytesIO(contenido))

    requeridas = VARIABLES_ORIGINALES + ["burnout_level"]
    faltantes = [c for c in requeridas if c not in df_eval.columns]
    if faltantes:
        return {"error": f"Faltan columnas requeridas: {faltantes}", "columnas_esperadas": requeridas}

    clases_validas = set(label_encoder.classes_)
    invalidas = sorted(set(df_eval["burnout_level"].unique()) - clases_validas)
    if invalidas:
        return {"error": f"Valores de burnout_level no reconocidos: {invalidas}. "
                          f"Deben ser exactamente: {sorted(clases_validas)}"}

    df_feat = _construir_features_batch(df_eval[VARIABLES_ORIGINALES])
    X_norm = pd.DataFrame(scaler.transform(df_feat), columns=feature_cols)
    y_true = label_encoder.transform(df_eval["burnout_level"])
    y_pred = modelo.predict(X_norm)
    y_proba = modelo.predict_proba(X_norm)

    n_clases = len(label_encoder.classes_)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clases))
    precision, recall, f1_clase, soporte = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_clases), zero_division=0
    )

    y_bin = label_binarize(y_true, classes=range(n_clases))
    curvas_roc = {}
    for i, clase in enumerate(label_encoder.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        curvas_roc[clase] = {
            "fpr": [round(float(v), 4) for v in fpr],
            "tpr": [round(float(v), 4) for v in tpr],
            "auc": round(float(auc(fpr, tpr)), 4),
        }

    return {
        "n_registros": len(df_eval),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1m), 4),
        "auc_roc_macro": round(float(auc_macro), 4),
        "clases": list(label_encoder.classes_),
        "matriz_confusion": cm.tolist(),
        "reporte_por_clase": [
            {"clase": c, "precision": round(float(p), 3), "recall": round(float(r), 3),
             "f1": round(float(f), 3), "soporte": int(s)}
            for c, p, r, f, s in zip(label_encoder.classes_, precision, recall, f1_clase, soporte)
        ],
        "curvas_roc": curvas_roc,
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
        "segmentacion_kmeans_disponible": HAY_KMEANS,
        "segmentos": list(NOMBRES_SEGMENTO.values()) if HAY_KMEANS else [],
        "red_neuronal_segunda_opinion": {"disponible": HAY_NN, **NN_INFO},
    }


@app.get("/model/compare")
def comparar_modelos():
    """
    Comparacion de los dos modelos desplegados (transparencia): LightGBM (boosting de
    arboles) vs Red Neuronal MLP, con su nivel de acuerdo en el conjunto de prueba.
    """
    if not METRICAS_MODELOS:
        return {"error": "El modelo desplegado no incluye metricas comparativas de la red neuronal."}
    return METRICAS_MODELOS


@app.get("/fairness")
def auditoria_equidad():
    """
    Auditoria de EQUIDAD / mitigacion de sesgos: rendimiento del modelo desglosado por
    subgrupos demograficos (edad y experiencia). Una brecha pequena entre subgrupos indica
    que el modelo no penaliza a ningun grupo. Calculado sobre el conjunto de prueba real.
    """
    if AUDITORIA_EQUIDAD is None:
        return {"error": "El modelo desplegado no incluye la auditoria de equidad."}
    return AUDITORIA_EQUIDAD


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


@app.get("/sample/evaluation")
def muestra_evaluacion(n: int = Query(50, ge=5, le=500)):
    """
    Descarga una muestra real (con burnout_level real) del dataset de entrenamiento,
    lista para probar el endpoint /evaluate sin tener que conseguir datos externos.
    """
    if not HAY_DATASET:
        return {"error": "Dataset de referencia no disponible en este despliegue."}
    muestra = df_entrenamiento.sample(min(n, len(df_entrenamiento)))
    return Response(
        content=muestra.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=muestra_evaluacion.csv"},
    )


@app.get("/health")
def salud():
    return {
        "status": "ok",
        "shap_disponible": HAY_SHAP,
        "dataset_referencia_disponible": HAY_DATASET,
        "segmentacion_kmeans_disponible": HAY_KMEANS,
        "red_neuronal_disponible": HAY_NN,
    }


@app.get("/")
def inicio():
    """Sirve el frontend del simulador What-If."""
    return FileResponse(os.path.join(RUTA_BASE, "static", "index.html"))
