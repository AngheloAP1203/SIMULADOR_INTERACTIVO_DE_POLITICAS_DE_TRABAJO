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
| `burnout_model.pkl` | Modelo LightGBM entrenado + `StandardScaler` + `LabelEncoder` + orden de variables |
| `burnout_mlp.pkl` | Modelo MLPClassifier (red neuronal) entrenado sobre el mismo scaler/features, generado por `train_mlp.py` |
| `train_mlp.py` | Script de entrenamiento del MLP, reutiliza el scaler y feature_cols de `burnout_model.pkl` |
| `api_burnout.py` | API REST con FastAPI: `POST /predict`, `POST /predict/batch`, `POST /evaluate`, `POST /optimize`, `POST /sensitivity`, `GET /model/info`, `GET /model/importance` |
| `static/index.html` | Frontend tipo dashboard: simulador What-If, comparador de escenarios, recomendaciones, comparación poblacional, segmentación K-Means, sensibilidad, predicción por lotes, evaluación con matriz de confusión/ROC, ficha técnica del modelo y comentario opcional con análisis de sentimiento |
| `df_burnout_procesado.csv` | Dataset real usado en el entrenamiento (7,000 registros), usado para percentiles y rangos de sensibilidad |
| `requirements.txt` | Dependencias de Python |
| `render.yaml` | Configuración del servicio web |

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"></path></svg> Sobre el modelo

- **Algoritmo principal:** LightGBM (clasificación multiclase, `class_weight='balanced'`)
- **Segundo modelo (red neuronal):** MLPClassifier (64, 32), comparado en vivo contra LightGBM en cada predicción vía `/predict` y en `/model/info`
- **Dataset:** Developer Burnout Prediction Dataset (Kaggle, 7 000 registros)
- **Variables:** 11 originales + 4 derivadas (`work_life_ratio`, `productivity_score`, `meeting_fatigue`, `recovery_index`)
- **Rendimiento (test):** ver métricas reales y verificables en vivo en `GET /model/info` (`comparacion_modelos`) o recalculándolas sobre datos propios en `POST /evaluate`
- **Hiperparámetros:** optimizados con `RandomizedSearchCV` (validación cruzada estratificada k=5) — los valores reales del modelo desplegado se leen en vivo desde `GET /model/info`, no están hardcodeados en este README
- **Interpretabilidad:** SHAP (TreeExplainer) integrado en el endpoint
- **Servicio cognitivo externo:** análisis de sentimiento (Hugging Face Inference API) sobre un comentario de texto libre opcional del desarrollador. Requiere la variable de entorno `HF_API_TOKEN` (gratuita en huggingface.co/settings/tokens); sin ella, la funcionalidad se desactiva con gracia (ver `GET /health`)

## <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px"><path d="M12 3v18"></path><path d="M5 7l-3 7a3 3 0 006 0z"></path><path d="M19 7l-3 7a3 3 0 006 0z"></path><path d="M3 7h18"></path><path d="M9 21h6"></path></svg> Uso responsable

Las predicciones de este sistema están orientadas a la **prevención y el bienestar**
del equipo. No deben usarse para sancionar, despedir ni discriminar a ninguna persona.
El modelo fue entrenado con datos sintéticos y sus resultados deben validarse con
datos reales antes de cualquier uso en producción.
