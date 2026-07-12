# 🧪 Evidencias de Pruebas — Simulador de Burnout

Registro de pruebas **reales** (nada simulado) ejecutadas sobre el modelo entrenado
`burnout_model.pkl` y sobre el despliegue en producción. Última ejecución: **11 de julio de 2026**.

---

## 1. Suite automatizada (pytest) — 15/15 pruebas superadas ✅

Ejecutada contra la API real con `TestClient` (peticiones HTTP reales en memoria, modelo real cargado).

```
pytest tests/ -v
```

| # | Prueba | Qué verifica | Resultado |
|---|---|---|---|
| 1 | `test_health_ok` | El servicio responde | ✅ |
| 2 | `test_model_info_expone_modelo_real` | LGBMClassifier real, 15 variables, 116 árboles | ✅ |
| 3 | `test_model_importance_stress_es_la_mas_importante` | `stress_level` domina la importancia nativa | ✅ |
| 4 | `test_predict_perfil_alto_riesgo_da_high` | Perfil sobrecargado → High, BRS ≥ 90 | ✅ |
| 5 | `test_predict_perfil_saludable_da_low` | Perfil saludable → Low, BRS ≤ 10 | ✅ |
| 6 | `test_predict_incluye_features_derivadas_correctas` | Ingeniería de características idéntica al entrenamiento | ✅ |
| 7 | `test_predict_valida_campos_faltantes` | Validación Pydantic (HTTP 422) | ✅ |
| 8 | `test_evaluate_con_muestra_real_supera_90_por_ciento` | Accuracy/F1 > 0.90 sobre 300 registros reales etiquetados | ✅ |
| 9 | `test_evaluate_rechaza_csv_sin_etiquetas` | Manejo de errores de entrada | ✅ |
| 10 | `test_predict_batch_procesa_varios_perfiles` | Lote CSV: predicciones + resumen coherentes | ✅ |
| 11 | `test_optimize_reduce_riesgo_de_perfil_critico` | El plan prescriptivo reduce el BRS real | ✅ |
| 12 | `test_optimize_perfil_sano_no_requiere_plan` | No prescribe intervenciones innecesarias | ✅ |
| 13 | `test_sensitivity_stress_es_creciente` | Dependencia parcial: más estrés ⇒ más riesgo (Low→High) | ✅ |
| 14 | `test_sensitivity_rechaza_variable_invalida` | Manejo de errores de parámetros | ✅ |
| 15 | `test_segmento_coherente_con_riesgo` | K-Means asigna segmentos coherentes con el riesgo | ✅ |

Resultado: **15 passed in 11.01s** (con SHAP real activo).

## 2. Pruebas del despliegue en producción (Render) ✅

Peticiones HTTP reales contra `https://simulador-interactivo-de-politicas-de.onrender.com`:

| Endpoint | Entrada | Respuesta real obtenida |
|---|---|---|
| `GET /health` | — | `{"status":"ok","shap_disponible":true,"dataset_referencia_disponible":true,"segmentacion_kmeans_disponible":true}` |
| `POST /predict` | Perfil sobrecargado (11 h trabajo, 4.5 h sueño, estrés 85) | **High**, BRS 99.6, segmento *Sobrecargado / alto riesgo*, SHAP top-1: `stress_level` (+4.11) |
| `POST /predict` | Perfil saludable (7 h trabajo, 8 h sueño, estrés 20) | **Low**, BRS 0.2, segmento *Equilibrado / bajo riesgo* |
| `POST /optimize` | Perfil sobrecargado, objetivo BRS ≤ 33 | Plan de 6 pasos, BRS 99.6 → **2.1**, objetivo alcanzado |
| `GET /model/info` | — | LGBMClassifier, 116 árboles, accuracy test 0.9819 |
| `POST /evaluate` | 200 registros reales con etiqueta verdadera | **Accuracy 0.995 · F1-macro 0.9948 · AUC 1.0** — matriz de confusión con **1 solo error de 200** |

> Nota: la primera petición tras inactividad tarda ~1 min (plan gratuito de Render "despierta" el servicio).

## 3. Validación del notebook completo

El notebook `machine.ipynb` (224 celdas) fue ejecutado íntegramente de punta a punta con
`nbclient` (motor real de Jupyter): **0 errores en 86 celdas de código** (~10 min), incluyendo
los 6 modelos supervisados, K-Means + DBSCAN, PCA/t-SNE, SHAP/LIME, bootstrap (1,000
remuestreos) y la exportación del `.pkl`.

## Cómo reproducir

```bash
pip install -r requirements.txt pytest
pytest tests/ -v                      # suite local (modelo real)
uvicorn api_burnout:app --reload      # servidor local + frontend en http://127.0.0.1:8000
```
