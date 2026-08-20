FROM python:3.13-slim

WORKDIR /app

ENV KERAS_BACKEND=torch
ENV DATASET_PATH=/app/data/dataset_rodoanel_10k.csv
ENV PREDICTION_LOG_PATH=/app/logs/predictions.jsonl
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Build a partir desta pasta (contexto = Modelo 1):
#   docker build -t temova-modelo1 .
# No Render: Root Directory vazio, Dockerfile Path = Dockerfile

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs /app/artifacts \
 && if [ ! -f artifacts/lstm_model.keras ]; then \
      echo "Artefatos ausentes — treinando (lento, só use se não versionou os pesos)."; \
      python train_model.py --epochs 50; \
    fi

VOLUME ["/app/logs"]

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
