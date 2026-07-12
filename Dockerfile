FROM python:3.11.9-slim

WORKDIR /app

# libgomp1: runtime OpenMP requerido por LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_burnout.py .
COPY burnout_model.pkl burnout_mlp.pkl ./
COPY df_burnout_procesado.csv .
COPY static/ ./static/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api_burnout:app --host 0.0.0.0 --port ${PORT:-8000}"]
