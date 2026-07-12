# <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><path d="M9 2v6.5L4 18a2 2 0 001.7 3h12.6a2 2 0 001.7-3l-5-9.5V2"></path><line x1="8" y1="2" x2="16" y2="2"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg> Evidencias de Pruebas — Simulador de Burnout

Registro de pruebas **reales** (nada simulado) ejecutadas sobre el modelo entrenado
`burnout_model.pkl` y sobre el despliegue en producción. Última ejecución: **11 de julio de 2026**.

---

## 1. Suite automatizada (pytest) — 22/22 pruebas superadas <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>

Ejecutada contra la API real con `TestClient` (peticiones HTTP reales en memoria, modelo real cargado).

```
pytest tests/ -v
```

| # | Prueba | Qué verifica | Resultado |
|---|---|---|---|
| 1 | `test_health_ok` | El servicio responde | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 2 | `test_model_info_expone_modelo_real` | LGBMClassifier real, 15 variables, 116 árboles | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 3 | `test_model_importance_stress_es_la_mas_importante` | `stress_level` domina la importancia nativa | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 4 | `test_predict_perfil_alto_riesgo_da_high` | Perfil sobrecargado → High, BRS ≥ 90 | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 5 | `test_predict_perfil_saludable_da_low` | Perfil saludable → Low, BRS ≤ 10 | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 6 | `test_predict_incluye_features_derivadas_correctas` | Ingeniería de características idéntica al entrenamiento | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 7 | `test_predict_valida_campos_faltantes` | Validación Pydantic (HTTP 422) | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 8 | `test_evaluate_con_muestra_real_supera_90_por_ciento` | Accuracy/F1 > 0.90 sobre 300 registros reales etiquetados | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 9 | `test_evaluate_rechaza_csv_sin_etiquetas` | Manejo de errores de entrada | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 10 | `test_predict_batch_procesa_varios_perfiles` | Lote CSV: predicciones + resumen coherentes | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 11 | `test_optimize_reduce_riesgo_de_perfil_critico` | El plan prescriptivo reduce el BRS real | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 12 | `test_optimize_perfil_sano_no_requiere_plan` | No prescribe intervenciones innecesarias | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 13 | `test_sensitivity_stress_es_creciente` | Dependencia parcial: más estrés ⇒ más riesgo (Low→High) | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 14 | `test_sensitivity_rechaza_variable_invalida` | Manejo de errores de parámetros | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 15 | `test_segmento_coherente_con_riesgo` | K-Means asigna segmentos coherentes con el riesgo | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 16 | `test_red_neuronal_coincide_en_perfiles_claros` | La red neuronal desplegada (2ª opinión) coincide con LightGBM en perfiles extremos | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 17 | `test_explicacion_en_lenguaje_natural_refleja_la_prediccion` | La explicación cognitiva menciona el nivel y los factores SHAP reales | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 18 | `test_cbi_items_expone_13_preguntas` | El test validado CBI expone sus 13 ítems reales + cita | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 19 | `test_cbi_maximo_da_severo` | Respuestas máximas → burnout global 100, categoría severo | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 20 | `test_cbi_minimo_sin_burnout` | Respuestas mínimas → burnout global 0, sin burnout | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 21 | `test_cbi_item_energia_es_invertido` | El ítem invertido del CBI se puntúa correctamente al revés | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |
| 22 | `test_cbi_valida_respuestas_invalidas` | Rechaza valores y cantidades de respuestas inválidas | <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> |

Resultado: **22 passed in 8.46s** (con SHAP real activo). Incluye el test clínico validado CBI
(evaluación real de burnout, sin Machine Learning ni datos simulados).

### Doble modelo en producción (verificado)
- Red neuronal MLP (64-32-16): **Accuracy 94.48 % / F1 0.944** en el mismo test held-out.
- **Acuerdo LightGBM ↔ Red neuronal: 95.33 %** de los 1,050 casos de prueba; las discrepancias
  se señalan en la API (`"coincide": false`) para revisión humana.

### Auditoría de equidad (verificada sobre el test set)
| Subgrupo | n | Accuracy | F1-macro |
|---|---|---|---|
| Edad 20-27 | 301 | 0.9734 | 0.9727 |
| Edad 28-36 | 413 | 0.9831 | 0.9835 |
| Edad 37-44 | 336 | 0.9881 | 0.9884 |
| Experiencia 0-4 | 244 | 0.9836 | 0.9831 |
| Experiencia 5-11 | 379 | 0.9894 | 0.9894 |
| Experiencia 12-19 | 427 | 0.9742 | 0.9746 |

Brecha máxima de F1 entre subgrupos: **≈ 0.016** — el modelo no penaliza a ningún grupo demográfico.

## 2. Pruebas del despliegue en producción (Render) <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>

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
