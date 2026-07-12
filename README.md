# 🧠 Simulador Interactivo de Políticas de Trabajo

Sistema predictivo de **Machine Learning** que clasifica el nivel de *burnout* de
desarrolladores de software en tres categorías (**Low / Medium / High**), expuesto como
API REST (FastAPI) con un **simulador What-If** web y explicaciones **SHAP** por predicción.

**Universidad Privada Antenor Orrego** · Ing. de Sistemas e Inteligencia Artificial · Machine Learning · 2026-10

**Integrantes:** Alcántara Pérez, Anghelo · Arenas Arriaga, Johan · Brunelli Watanabe, Valeria · Cabrera Mendoza, Fabio · Campos Acevedo, Gianfranco

---

## 📦 Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `burnout_model.pkl` | Modelo LightGBM entrenado + `StandardScaler` + `LabelEncoder` + orden de variables |
| `api_burnout.py` | API REST con FastAPI: `POST /predict`, `POST /predict/batch`, `POST /evaluate`, `POST /optimize`, `POST /sensitivity`, `GET /model/info`, `GET /model/importance` |
| `static/index.html` | Frontend tipo dashboard: simulador What-If, comparador de escenarios, recomendaciones, comparación poblacional, segmentación K-Means, sensibilidad, predicción por lotes, evaluación con matriz de confusión/ROC y ficha técnica del modelo |
| `df_burnout_procesado.csv` | Dataset real usado en el entrenamiento (7,000 registros), usado para percentiles y rangos de sensibilidad |
| `requirements.txt` | Dependencias de Python |
| `render.yaml` | Configuración del servicio web |

## 🤖 Sobre el modelo

- **Algoritmo:** LightGBM (clasificación multiclase, `class_weight='balanced'`)
- **Dataset:** Developer Burnout Prediction Dataset (Kaggle, 7 000 registros)
- **Variables:** 11 originales + 4 derivadas (`work_life_ratio`, `productivity_score`, `meeting_fatigue`, `recovery_index`)
- **Rendimiento (test):** Accuracy 98.19 % · F1-macro 0.982 · AUC-ROC 0.9953
- **Hiperparámetros:** optimizados con `RandomizedSearchCV` (validación cruzada estratificada k=5)
- **Interpretabilidad:** SHAP (TreeExplainer) integrado en el endpoint

## 🧪 Pruebas

El proyecto incluye una **suite de 15 pruebas automatizadas** (`tests/test_api.py`, pytest) que
ejecutan el modelo real y verifican predicciones, evaluación con datos etiquetados, optimizador,
sensibilidad y segmentación. Resultados y evidencias del despliegue en producción: ver [PRUEBAS.md](PRUEBAS.md).

```bash
pytest tests/ -v    # 15 passed
```

## 📋 Evidencias verificables (mapeo de competencias)

| Requisito | Evidencia en este proyecto |
|---|---|
| Preprocesamiento de datos | Imputación mediana/moda, winsorización, `StandardScaler` (notebook secc. 4; mismo pipeline en `api_burnout.py`) |
| Selección de características | 11 originales + 4 derivadas justificadas; ranking de importancia nativa + SHAP (`GET /model/importance`) |
| Redes neuronales | MLP con TensorFlow/Keras (3 capas, ReLU, dropout, early stopping) comparada contra 5 modelos más (notebook secc. 5) |
| Entrenamiento + validación con métricas | `RandomizedSearchCV` + CV estratificada k=5; Accuracy 98.19 %, F1 0.982, AUC 0.9953 en test held-out; **IC 95 % por bootstrap (1,000 remuestreos)** |
| Despliegue funcional | API FastAPI + dashboard en Render (este repo); endpoint `/evaluate` recalcula métricas en vivo con sklearn |
| Evidencias verificables de funcionamiento | [PRUEBAS.md](PRUEBAS.md): 15/15 pruebas pytest + pruebas HTTP reales en producción |
| Transparencia | SHAP (global + waterfall por predicción, expuesto en la API) + LIME (contraste), coeficientes de Regresión Logística |
| Mitigación de sesgos | `class_weight='balanced'` en todos los modelos (desbalance 52/26/23); F1-macro como métrica (pondera igual las 3 clases); auditoría de variables sensibles con SHAP; matriz de confusión por clase sin degradar minorías |
| Validación cruzada | Estratificada k=5 (F1 0.9845 ± 0.0034) + bootstrap para intervalos de confianza |
| Optimización | Búsqueda de hiperparámetros (25 comb. × 5 pliegues) + early stopping |
| Impacto social | Análisis poblacional con ROI estimado (~$4.1M/año bajo supuestos declarados); optimizador prescriptivo de políticas de bienestar; uso preventivo y no punitivo (secc. 11 del notebook) |

## ⚖️ Uso responsable

Las predicciones de este sistema están orientadas a la **prevención y el bienestar**
del equipo. No deben usarse para sancionar, despedir ni discriminar a ninguna persona.
El modelo fue entrenado con datos sintéticos y sus resultados deben validarse con
datos reales antes de cualquier uso en producción.
