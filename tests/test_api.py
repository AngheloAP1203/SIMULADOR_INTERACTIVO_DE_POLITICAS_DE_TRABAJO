"""
Suite de pruebas REALES del Simulador de Burnout (pytest).

Todas las pruebas ejecutan el modelo LightGBM realmente entrenado (burnout_model.pkl)
a traves de la API FastAPI con TestClient (peticiones HTTP reales en memoria).
Nada es simulado: las metricas se calculan con scikit-learn sobre datos reales
del dataset de entrenamiento.

Ejecutar:  pytest tests/ -v
"""
import io
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Permite importar api_burnout desde la raiz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api_burnout
from api_burnout import app

cliente = TestClient(app)

PERFIL_ALTO_RIESGO = {
    "age": 30, "experience_years": 5, "daily_work_hours": 11, "sleep_hours": 4.5,
    "caffeine_intake": 5, "bugs_per_day": 8, "commits_per_day": 12, "meetings_per_day": 8,
    "screen_time": 13, "exercise_hours": 0.1, "stress_level": 85,
}
PERFIL_SALUDABLE = {
    "age": 30, "experience_years": 8, "daily_work_hours": 7, "sleep_hours": 8,
    "caffeine_intake": 1, "bugs_per_day": 5, "commits_per_day": 10, "meetings_per_day": 2,
    "screen_time": 8, "exercise_hours": 1.5, "stress_level": 20,
}


# ---------------------------------------------------------------------------
# 1. Salud y ficha del modelo
# ---------------------------------------------------------------------------
def test_health_ok():
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info_expone_modelo_real():
    r = cliente.get("/model/info")
    assert r.status_code == 200
    d = r.json()
    assert d["algoritmo"] == "LGBMClassifier"
    assert d["n_variables"] == 15                      # 11 originales + 4 derivadas
    assert sorted(d["clases"]) == ["High", "Low", "Medium"]
    assert d["arboles_usados"] > 0                     # entrenado de verdad (early stopping)
    assert 0.9 < d["metricas_test"]["accuracy"] <= 1.0


def test_model_importance_stress_es_la_mas_importante():
    r = cliente.get("/model/importance")
    assert r.status_code == 200
    imp = r.json()["importancias"]
    assert len(imp) == 15
    # Hallazgo documentado en el notebook: stress_level domina la importancia nativa
    assert imp[0]["variable"] == "stress_level"


# ---------------------------------------------------------------------------
# 2. Prediccion individual (coherencia del modelo real)
# ---------------------------------------------------------------------------
def test_predict_perfil_alto_riesgo_da_high():
    r = cliente.post("/predict", json=PERFIL_ALTO_RIESGO)
    assert r.status_code == 200
    d = r.json()
    assert d["burnout_predicho"] == "High"
    assert d["burnout_risk_score"] >= 90
    assert abs(sum(d["probabilidades"].values()) - 100) < 0.5   # probabilidades suman 100%


def test_predict_perfil_saludable_da_low():
    r = cliente.post("/predict", json=PERFIL_SALUDABLE)
    assert r.status_code == 200
    d = r.json()
    assert d["burnout_predicho"] == "Low"
    assert d["burnout_risk_score"] <= 10


def test_predict_incluye_features_derivadas_correctas():
    r = cliente.post("/predict", json=PERFIL_ALTO_RIESGO)
    deriv = r.json()["features_derivadas"]
    # Misma ingenieria de caracteristicas que el entrenamiento
    assert deriv["work_life_ratio"] == round(11 / 4.5, 2)
    assert deriv["productivity_score"] == round(12 * 0.6 + 8 * 0.4, 2)
    assert deriv["meeting_fatigue"] == round(8 * 11, 2)
    assert deriv["recovery_index"] == round(4.5 + 0.1, 2)


def test_predict_valida_campos_faltantes():
    r = cliente.post("/predict", json={"age": 30})   # faltan 10 campos
    assert r.status_code == 422                       # validacion de Pydantic


# ---------------------------------------------------------------------------
# 3. Evaluacion REAL contra datos etiquetados del dataset
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not api_burnout.HAY_DATASET, reason="dataset de referencia no disponible")
def test_evaluate_con_muestra_real_supera_90_por_ciento():
    muestra = api_burnout.df_entrenamiento.sample(300, random_state=7)
    csv_bytes = muestra.to_csv(index=False).encode()
    r = cliente.post("/evaluate", files={"archivo": ("m.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    d = r.json()
    assert d["n_registros"] == 300
    # El modelo real debe evaluar con metricas altas sobre datos reales etiquetados
    assert d["accuracy"] > 0.90
    assert d["f1_macro"] > 0.90
    assert d["auc_roc_macro"] > 0.95
    # La matriz de confusion 3x3 debe sumar el total de registros
    assert sum(sum(fila) for fila in d["matriz_confusion"]) == 300


def test_evaluate_rechaza_csv_sin_etiquetas():
    csv_malo = "age\n30\n".encode()
    r = cliente.post("/evaluate", files={"archivo": ("malo.csv", csv_malo, "text/csv")})
    assert "error" in r.json()


# ---------------------------------------------------------------------------
# 4. Prediccion por lotes
# ---------------------------------------------------------------------------
def test_predict_batch_procesa_varios_perfiles():
    df = pd.DataFrame([PERFIL_ALTO_RIESGO, PERFIL_SALUDABLE])
    csv_bytes = df.to_csv(index=False).encode()
    r = cliente.post("/predict/batch", files={"archivo": ("equipo.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    d = r.json()
    assert d["total_registros"] == 2
    assert d["resultados"][0]["burnout_predicho"] == "High"
    assert d["resultados"][1]["burnout_predicho"] == "Low"
    assert d["resumen"]["en_riesgo_alto"] == 1


# ---------------------------------------------------------------------------
# 5. Optimizador prescriptivo (mitigacion accionable)
# ---------------------------------------------------------------------------
def test_optimize_reduce_riesgo_de_perfil_critico():
    r = cliente.post("/optimize?objetivo_brs=33&costo_maximo=25", json=PERFIL_ALTO_RIESGO)
    assert r.status_code == 200
    d = r.json()
    assert d["brs_final"] < d["brs_inicial"]          # el plan reduce el riesgo real
    assert d["brs_final"] <= 33 or d["costo_total"] >= 25
    assert len(d["plan"]) > 0


def test_optimize_perfil_sano_no_requiere_plan():
    r = cliente.post("/optimize", json=PERFIL_SALUDABLE)
    d = r.json()
    assert d["plan"] == []                             # ya esta bajo el objetivo


# ---------------------------------------------------------------------------
# 6. Analisis de sensibilidad (dependencia parcial real)
# ---------------------------------------------------------------------------
def test_sensitivity_stress_es_creciente():
    r = cliente.post("/sensitivity?variable=stress_level&n_puntos=15", json=PERFIL_SALUDABLE)
    assert r.status_code == 200
    curva = r.json()["curva"]
    assert len(curva) == 15
    brs = [p["burnout_risk_score"] for p in curva]
    clases = [p["clase_predicha"] for p in curva]
    # Tendencia creciente global: el modelo de arboles produce una funcion escalonada
    # (Low -> Medium -> High) con micro-oscilaciones de ~2 pts dentro de cada escalon,
    # por eso se tolera hasta 3 pts de caida local pero se exige el salto total.
    assert all(brs[i] <= brs[i + 1] + 3.0 for i in range(len(brs) - 1))
    assert brs[-1] - brs[0] > 50                       # el barrido completo cruza de Low a High
    assert clases[0] == "Low" and clases[-1] == "High"


def test_sensitivity_rechaza_variable_invalida():
    r = cliente.post("/sensitivity?variable=no_existe", json=PERFIL_SALUDABLE)
    assert "error" in r.json()


# ---------------------------------------------------------------------------
# 7. Segmentacion no supervisada (K-Means real)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not api_burnout.HAY_KMEANS, reason="segmentacion no disponible")
def test_segmento_coherente_con_riesgo():
    alto = cliente.post("/predict", json=PERFIL_ALTO_RIESGO).json()["segmento"]
    sano = cliente.post("/predict", json=PERFIL_SALUDABLE).json()["segmento"]
    assert alto["segmento_nombre"] == "Sobrecargado / alto riesgo"
    assert sano["segmento_nombre"] == "Equilibrado / bajo riesgo"


# ---------------------------------------------------------------------------
# 8. Red neuronal en produccion (segunda opinion) y explicacion cognitiva
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not api_burnout.HAY_NN, reason="red neuronal no disponible")
def test_red_neuronal_coincide_en_perfiles_claros():
    # En perfiles extremos, LightGBM y la red neuronal (familias de modelos distintas)
    # deben dar la MISMA clase: evidencia real de doble modelo funcional en produccion.
    for perfil, esperado in [(PERFIL_ALTO_RIESGO, "High"), (PERFIL_SALUDABLE, "Low")]:
        d = cliente.post("/predict", json=perfil).json()
        so = d["segunda_opinion_red_neuronal"]
        assert so["burnout_predicho"] == esperado
        assert so["coincide"] is True


def test_explicacion_en_lenguaje_natural_refleja_la_prediccion():
    d = cliente.post("/predict", json=PERFIL_ALTO_RIESGO).json()
    texto = d["explicacion_texto"]
    assert "ALTO" in texto                                # menciona el nivel real predicho
    assert "estrés" in texto or "estres" in texto          # el factor SHAP dominante real
    assert len(texto) > 80                                 # explicacion completa, no un stub


# ---------------------------------------------------------------------------
# 9. Copenhagen Burnout Inventory (CBI) - autoevaluacion REAL validada (sin ML)
# ---------------------------------------------------------------------------
def test_cbi_items_expone_13_preguntas():
    d = cliente.get("/cbi/items").json()
    assert d["n_items"] == 13
    assert len(d["preguntas"]) == 13
    assert [o["valor"] for o in d["opciones"]] == [100, 75, 50, 25, 0]
    assert "Kristensen" in d["citacion"]                   # cita el instrumento real


def test_cbi_maximo_da_severo():
    # 12 items de agotamiento al maximo + item de energia (invertido) en 0 = burnout total
    r = cliente.post("/cbi", json={"respuestas": [100] * 12 + [0]}).json()
    assert r["burnout_global"] == 100.0
    assert r["categoria"] == "Burnout severo"
    assert r["burnout_presente"] is True


def test_cbi_minimo_sin_burnout():
    r = cliente.post("/cbi", json={"respuestas": [0] * 12 + [100]}).json()
    assert r["burnout_global"] == 0.0
    assert r["burnout_presente"] is False


def test_cbi_item_energia_es_invertido():
    # Mas energia para familia/amigos DEBE reducir el burnout laboral (item invertido real del CBI)
    base = [0] * 13
    sin_energia = cliente.post("/cbi", json={"respuestas": base}).json()["burnout_laboral"]
    base[12] = 100
    con_energia = cliente.post("/cbi", json={"respuestas": base}).json()["burnout_laboral"]
    assert con_energia < sin_energia


def test_cbi_valida_respuestas_invalidas():
    assert "error" in cliente.post("/cbi", json={"respuestas": [42] * 13}).json()   # valor no permitido
    assert "error" in cliente.post("/cbi", json={"respuestas": [0] * 5}).json()      # cantidad incorrecta
