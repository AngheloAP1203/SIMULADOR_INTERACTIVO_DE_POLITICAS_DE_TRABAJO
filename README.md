# <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg> Simulador Interactivo de Políticas de Trabajo

Sistema predictivo de **Machine Learning** que clasifica el nivel de *burnout* de
desarrolladores de software en tres categorías (**Low / Medium / High**), expuesto como
API REST (FastAPI) con un **simulador What-If** web y explicaciones **SHAP** por predicción.

**Universidad Privada Antenor Orrego** · Ing. de Sistemas e Inteligencia Artificial · Machine Learning · 2026-10

**Integrantes:** Alcántara Pérez, Anghelo · Arenas Arriaga, Johan · Brunelli Watanabe, Valeria · Cabrera Mendoza, Fabio · Campos Acevedo, Gianfranco

---

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><path d="M12 2 2 7l10 5 10-5-10-5z"></path><path d="M2 17l10 5 10-5"></path><path d="M2 12l10 5 10-5"></path></svg> Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `burnout_model.pkl` | LightGBM + red neuronal MLP (64-32-16, "segunda opinión") + `StandardScaler` + `LabelEncoder` + auditoría de equidad, en un solo artefacto |
| `burnout_mlp.pkl` | Segundo modelo MLP independiente, generado por `train_mlp.py`, comparado en vivo en `/predict` y `/model/info` |
| `train_mlp.py` | Script de entrenamiento del MLP independiente, reutiliza el scaler y feature_cols de `burnout_model.pkl` |
| `tests/test_api.py` | Suite de pruebas automatizadas (pytest) contra el modelo real y el test CBI |
| `api_burnout.py` | API REST con FastAPI: `POST /cbi` (test validado), `POST /predict`, `POST /predict/batch`, `POST /evaluate`, `POST /optimize`, `POST /sensitivity`, `GET /model/info`, `GET /model/compare`, `GET /fairness`, `GET /model/importance` |
| `static/index.html` | Frontend tipo dashboard: simulador What-If, comparador de escenarios, recomendaciones, comparación poblacional, segmentación K-Means, sensibilidad, predicción por lotes, evaluación con matriz de confusión/ROC, ficha técnica del modelo, comentario opcional con análisis de sentimiento, y autoevaluación real con el CBI |
| `df_burnout_procesado.csv` | Dataset real usado en el entrenamiento (7,000 registros), usado para percentiles y rangos de sensibilidad |
| `requirements.txt` | Dependencias de Python |
| `render.yaml` | Configuración del servicio web |

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg> Dos capas: evaluación REAL (CBI) + simulador predictivo (ML)

El sistema combina, con total transparencia, dos componentes de distinta naturaleza:

1. **Autoevaluación real con el Copenhagen Burnout Inventory (CBI)** — instrumento clínico **validado y de
   dominio público** (Kristensen TS, Borritz M, Villadsen E, Christensen KB. *Work & Stress*.
   2005;19(3):192-207). Su puntuación es una **fórmula publicada y determinística**: **no usa Machine Learning
   ni datos sintéticos**. Una persona real responde 13 preguntas validadas y obtiene su nivel real de burnout
   (endpoint `POST /cbi`). *Es la parte del sistema que sirve, hoy, a una persona real.*
2. **Simulador predictivo (Machine Learning)** — el modelo LightGBM + dos redes neuronales de comparación,
   entrenado con el dataset *sintético* de Kaggle. Se declara honestamente como **prototipo** (pendiente de
   validación con datos reales) y sirve para explorar **qué cambios de hábitos/trabajo reducirían el riesgo**
   (What-If, SHAP, optimizador).

El CBI dice *dónde estás* (real y validado); el simulador explora *qué hacer* (predictivo). Esta separación
explícita es una decisión de **transparencia y honestidad** deliberada.

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"></path></svg> Sobre el modelo

- **Algoritmo principal:** LightGBM (clasificación multiclase, `class_weight='balanced'`)
- **Segundo modelo (red neuronal independiente):** MLPClassifier (64, 32) en `burnout_mlp.pkl`, comparado en vivo contra LightGBM en cada predicción vía `/predict` y en `/model/info`
- **Segunda opinión (red neuronal consolidada):** MLP (64-32-16) empaquetada dentro de `burnout_model.pkl`, expuesta en `/model/compare` y usada para la auditoría de equidad en `/fairness`
- **Dataset:** Developer Burnout Prediction Dataset (Kaggle, 7 000 registros)
- **Variables:** 11 originales + 4 derivadas (`work_life_ratio`, `productivity_score`, `meeting_fatigue`, `recovery_index`)
- **Rendimiento (test):** ver métricas reales y verificables en vivo en `GET /model/info` (`comparacion_modelos`) o recalculándolas sobre datos propios en `POST /evaluate`
- **Hiperparámetros:** optimizados con `RandomizedSearchCV` (validación cruzada estratificada k=5) — los valores reales del modelo desplegado se leen en vivo desde `GET /model/info`, no están hardcodeados en este README
- **Interpretabilidad:** SHAP (TreeExplainer) integrado en el endpoint
- **Servicio cognitivo externo:** análisis de sentimiento (Hugging Face Inference API) sobre un comentario de texto libre opcional del desarrollador. Requiere la variable de entorno `HF_API_TOKEN` (gratuita en huggingface.co/settings/tokens); sin ella, la funcionalidad se desactiva con gracia (ver `GET /health`)

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><path d="M9 2v6.5L4 18a2 2 0 001.7 3h12.6a2 2 0 001.7-3l-5-9.5V2"></path><line x1="8" y1="2" x2="16" y2="2"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg> Pruebas

El proyecto incluye una **suite de 15 pruebas automatizadas** (`tests/test_api.py`, pytest) que
ejecutan el modelo real y verifican predicciones, evaluación con datos etiquetados, optimizador,
sensibilidad y segmentación. Resultados y evidencias del despliegue en producción: ver [PRUEBAS.md](PRUEBAS.md).

```bash
pytest tests/ -v    # 15 passed
```

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg> Evidencias verificables (mapeo de competencias)

| Requisito | Evidencia en este proyecto |
|---|---|
| Preprocesamiento de datos | Imputación mediana/moda, winsorización, `StandardScaler` (notebook secc. 4; mismo pipeline en `api_burnout.py`) |
| Selección de características | Técnica formal de **información mutua** (`mutual_info_classif`, notebook secc. 12.3b) + correlación + importancia nativa y SHAP; 11 originales + 4 derivadas justificadas (`GET /model/importance`) |
| Redes neuronales **integradas en el sistema funcional** | MLP con TensorFlow/Keras evaluada contra 5 modelos más (notebook secc. 5) **y desplegada en producción** (`nn_model.pkl`): cada `/predict` devuelve la segunda opinión de la red neuronal y marca discrepancias para revisión humana |
| Servicios cognitivos | El sistema desplegado expone servicios de IA consumibles por API: **decisión** (`/predict`, `/predict/batch`), **explicación en lenguaje natural** generada desde SHAP (`explicacion_texto` en cada respuesta), **recomendación prescriptiva** (`/optimize`) y **evaluación** (`/evaluate`) |
| Entrenamiento + validación con métricas | `RandomizedSearchCV` + CV estratificada k=5; Accuracy 98.19 %, F1 0.982, AUC 0.9953 en test held-out; **IC 95 % por bootstrap (1,000 remuestreos)** |
| Despliegue funcional | API FastAPI + dashboard en Render (este repo); endpoint `/evaluate` recalcula métricas en vivo con sklearn |
| Evidencias verificables de funcionamiento | [PRUEBAS.md](PRUEBAS.md): 17/17 pruebas pytest + pruebas HTTP reales en producción |
| Transparencia | SHAP (global + waterfall, expuesto en la API) + LIME (contraste) + coeficientes de Regresión Logística + **explicación en lenguaje natural por predicción** + **doble modelo con detección de discrepancias** |
| Mitigación de sesgos (registrada) | `class_weight='balanced'` en los 6 modelos; F1-macro como métrica; **auditoría de equidad por subgrupos de edad y experiencia (brecha máx. F1 ≈ 0.016, notebook secc. 8.6)**; información mutua ≈ 0 de atributos demográficos; registro formal en notebook secc. 11.1 |
| Validación cruzada | Estratificada k=5 (F1 0.9845 ± 0.0034) + bootstrap para intervalos de confianza |
| Optimización | Búsqueda de hiperparámetros (25 comb. × 5 pliegues) + early stopping |
| Impacto social | Análisis poblacional con ROI estimado (~$4.1M/año bajo supuestos declarados); optimizador prescriptivo de políticas de bienestar; uso preventivo y no punitivo (secc. 11 del notebook) |

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><path d="M12 3v18"></path><path d="M5 7l-3 7a3 3 0 006 0z"></path><path d="M19 7l-3 7a3 3 0 006 0z"></path><path d="M3 7h18"></path><path d="M9 21h6"></path></svg> Uso responsable

Las predicciones de este sistema están orientadas a la **prevención y el bienestar**
del equipo. No deben usarse para sancionar, despedir ni discriminar a ninguna persona.
El modelo fue entrenado con datos sintéticos y sus resultados deben validarse con
datos reales antes de cualquier uso en producción.
