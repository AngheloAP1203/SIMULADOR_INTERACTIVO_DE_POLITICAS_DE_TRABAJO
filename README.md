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
| `api_burnout.py` | API REST con FastAPI: `POST /predict` (predicción + BRS + explicación SHAP) y `POST /optimize` (recomendador de políticas de bienestar) |
| `static/index.html` | Frontend tipo dashboard: simulador What-If con gráficas interactivas, comparador de escenarios y recomendaciones |
| `requirements.txt` | Dependencias de Python |
| `render.yaml` | Configuración del servicio web |

## 🤖 Sobre el modelo

- **Algoritmo:** LightGBM (clasificación multiclase, `class_weight='balanced'`)
- **Dataset:** Developer Burnout Prediction Dataset (Kaggle, 7 000 registros)
- **Variables:** 11 originales + 4 derivadas (`work_life_ratio`, `productivity_score`, `meeting_fatigue`, `recovery_index`)
- **Rendimiento (test):** Accuracy 98.19 % · F1-macro 0.982 · AUC-ROC 0.9953
- **Hiperparámetros:** optimizados con `RandomizedSearchCV` (validación cruzada estratificada k=5)
- **Interpretabilidad:** SHAP (TreeExplainer) integrado en el endpoint

## ⚖️ Uso responsable

Las predicciones de este sistema están orientadas a la **prevención y el bienestar**
del equipo. No deben usarse para sancionar, despedir ni discriminar a ninguna persona.
El modelo fue entrenado con datos sintéticos y sus resultados deben validarse con
datos reales antes de cualquier uso en producción.
