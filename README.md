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
| `api_burnout.py` | API REST con FastAPI: `POST /predict` (predicción + BRS + explicación SHAP) |
| `static/index.html` | Frontend del simulador What-If (sliders interactivos) |
| `requirements.txt` | Dependencias de Python |
| `render.yaml` | Configuración de despliegue automático en Render.com |

## 🚀 Ejecutar en local

```bash
pip install -r requirements.txt
uvicorn api_burnout:app --reload
```

Abrir <http://127.0.0.1:8000> → simulador web.
Documentación interactiva de la API en <http://127.0.0.1:8000/docs>.

## 🌐 Desplegar gratis en Render.com

1. Crear cuenta gratuita en <https://render.com> (no pide tarjeta).
2. **New → Web Service** → conectar este repositorio de GitHub.
3. Render detecta `render.yaml` automáticamente. Si lo pide manualmente:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api_burnout:app --host 0.0.0.0 --port $PORT`
4. **Deploy** → obtendrás una URL pública tipo `https://simulador-burnout.onrender.com`.

> ⚠️ En el plan gratuito, el servicio "duerme" tras 15 min sin uso; la primera
> petición después puede tardar ~1 min en despertar.

## 🔌 Uso de la API

```bash
curl -X POST https://TU-URL.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30, "experience_years": 5, "daily_work_hours": 11,
    "sleep_hours": 4.5, "caffeine_intake": 5, "bugs_per_day": 8,
    "commits_per_day": 12, "meetings_per_day": 8, "screen_time": 13,
    "exercise_hours": 0.1, "stress_level": 85
  }'
```

Respuesta:

```json
{
  "burnout_predicho": "High",
  "burnout_risk_score": 99.3,
  "probabilidades": {"High": 99.3, "Low": 0.1, "Medium": 0.7},
  "explicacion_shap_top3": [
    {"variable": "stress_level", "contribucion": 4.1146},
    {"variable": "work_life_ratio", "contribucion": 0.2103},
    {"variable": "meeting_fatigue", "contribucion": 0.2043}
  ]
}
```

## 🤖 Sobre el modelo

- **Algoritmo:** LightGBM (clasificación multiclase, `class_weight='balanced'`)
- **Dataset:** Developer Burnout Prediction Dataset (Kaggle, 7 000 registros)
- **Variables:** 11 originales + 4 derivadas (`work_life_ratio`, `productivity_score`, `meeting_fatigue`, `recovery_index`)
- **Rendimiento (test):** Accuracy 98.19 % · F1-macro 0.982 · AUC-ROC 0.9953
- **Hiperparámetros:** optimizados con `RandomizedSearchCV` (validación cruzada estratificada k=5)
- **Interpretabilidad:** SHAP (TreeExplainer) integrado en el endpoint

El notebook completo con el pipeline (EDA, limpieza, comparación de 5 modelos,
entrenamiento, evaluación, SHAP/LIME y despliegue) se encuentra en Kaggle.
