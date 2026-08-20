from pathlib import Path
import json
import time
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.ops_store import OPEN_STATUSES, OpsStore
from api.schemas import (
    DispatchCreateRequest,
    DispatchPatchRequest,
    HealthResponse,
    PresenceRequest,
    PredictionLogsResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionStatsResponse,
    PredictionsSyncRequest,
)
from ml.config import DATASET_PATH, LOOK_BACK, LIMITE_ALERTA_CM, LIMITE_PODA_CM, LOG_DIR, PREDICTION_LOG_PATH
from ml.prediction_logger import PredictionLogger
from ml.predictor import GrassGrowthPredictor

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "samples"

NIVEIS_ALERTA = ("verde", "amarelo", "vermelho", "laranja")

app = FastAPI(
    title="API MLOps - Crescimento da Grama (Modelo 1)",
    description=(
        "Serviço de inferência do modelo LSTM de série temporal para previsão "
        "do crescimento acumulado da vegetação (cm) no trecho do Rodoanel. "
        f"Limite operacional de poda: {LIMITE_PODA_CM} cm. "
        f"Alerta preventivo a partir de {LIMITE_ALERTA_CM} cm."
    ),
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: GrassGrowthPredictor | None = None
prediction_logger: PredictionLogger | None = None
ops_store: OpsStore | None = None


@app.on_event("startup")
def load_model_on_startup() -> None:
    global predictor, prediction_logger, ops_store
    predictor = GrassGrowthPredictor()
    prediction_logger = PredictionLogger(PREDICTION_LOG_PATH)
    ops_store = OpsStore(LOG_DIR / "ops_state.json")


@app.get("/health", response_model=HealthResponse, tags=["MLOps"])
def health_check() -> HealthResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    return HealthResponse(
        status="ok",
        model_version=predictor.model_version,
        look_back=predictor.look_back,
    )


@app.get("/model/info", tags=["MLOps"])
def model_info() -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    return {
        "modelo": "LSTM (2 camadas + Dropout)",
        "alvo": "Crescimento_Acumulado_cm",
        "limite_alerta_cm": LIMITE_ALERTA_CM,
        "limite_poda_cm": LIMITE_PODA_CM,
        "look_back": predictor.look_back,
        "versao": predictor.model_version,
        "features": len(predictor.artifacts["feature_columns"]),
        "descricao": (
            f"Previsão de crescimento acumulado da grama em centímetros. "
            f"Alerta preventivo a partir de {LIMITE_ALERTA_CM} cm; "
            f"poda recomendada a partir de {LIMITE_PODA_CM} cm."
        ),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inferência"])
def predict_growth(
    request: PredictionRequest,
    x_source_system: str | None = Header(default=None, alias="X-Source-System"),
) -> PredictionResponse:
    if predictor is None or prediction_logger is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")

    records = [record.model_dump(by_alias=True) for record in request.records]
    for record in records:
        record["Data"] = str(record["Data"])
        record["Data_Ultima_Poda"] = str(record["Data_Ultima_Poda"])

    id_ponto = records[-1].get("ID_Ponto", "desconhecido")
    data_referencia = records[-1].get("Data", "")

    started_at = time.perf_counter()
    try:
        result = predictor.predict(records)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {exc}") from exc

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

    prediction_id = prediction_logger.log(
        {
            "id_ponto": id_ponto,
            "data_referencia": data_referencia,
            "crescimento_acumulado_cm": result["crescimento_acumulado_cm"],
            "nivel_alerta": result["nivel_alerta"],
            "nivel_label": result["nivel_label"],
            "alerta_preventivo": result["alerta_preventivo"],
            "acima_limite_poda": result["acima_limite_poda"],
            "distancia_limite_cm": result["distancia_limite_cm"],
            "distancia_alerta_cm": result["distancia_alerta_cm"],
            "model_version": result["model_version"],
            "latency_ms": latency_ms,
            "source_system": x_source_system or "api",
        }
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        id_ponto=id_ponto,
        data_referencia=data_referencia,
        latency_ms=latency_ms,
        **result,
    )


@app.get("/monitoring/logs", response_model=PredictionLogsResponse, tags=["Monitoramento"])
def get_prediction_logs(
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    id_ponto: Annotated[str | None, Query(examples=["RODO_KM_01"])] = None,
    nivel_alerta: Annotated[str | None, Query(examples=["verde"])] = None,
    since: Annotated[datetime | None, Query(description="ISO 8601 — início do intervalo")] = None,
    until: Annotated[datetime | None, Query(description="ISO 8601 — fim do intervalo")] = None,
) -> PredictionLogsResponse:
    """Retorna histórico de predições para consumo por dashboards externos."""
    if prediction_logger is None:
        raise HTTPException(status_code=503, detail="Logger ainda não inicializado.")

    logs = prediction_logger.query(
        limit=limit,
        id_ponto=id_ponto,
        nivel_alerta=nivel_alerta,
        since=since,
        until=until,
    )
    return PredictionLogsResponse(total=len(logs), limit=limit, logs=logs)


@app.get("/monitoring/stats", response_model=PredictionStatsResponse, tags=["Monitoramento"])
def get_prediction_stats(
    window_hours: Annotated[int, Query(ge=1, le=168)] = 24,
) -> PredictionStatsResponse:
    """Agregados de predições para painéis de monitoramento."""
    if prediction_logger is None:
        raise HTTPException(status_code=503, detail="Logger ainda não inicializado.")

    return PredictionStatsResponse(**prediction_logger.stats(window_hours=window_hours))


@app.get("/sample-alerta/{nivel}", tags=["Inferência"])
def get_sample_by_alert(nivel: str) -> dict:
    """Retorna amostra calibrada para testar cada nível de alerta."""
    nivel = nivel.lower()
    if nivel not in NIVEIS_ALERTA:
        raise HTTPException(
            status_code=404,
            detail=f"Nível inválido. Use: {', '.join(NIVEIS_ALERTA)}",
        )

    sample_path = BASE_DIR / f"sample_alerta_{nivel}.json"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Amostra '{nivel}' não encontrada.")

    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    else:
        raise HTTPException(
            status_code=500,
            detail="Formato de amostra inválido. Esperado array de registros ou objeto com chave 'records'.",
        )

    manifest_path = SAMPLES_DIR / "manifest.json"
    meta = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta = next((a for a in manifest["amostras"] if a["nivel"] == nivel), {})

    return {
        "nivel_alerta": nivel,
        "label": meta.get("label", nivel),
        "faixa_cm": meta.get("faixa_cm"),
        "predicao_esperada_cm": meta.get("predicao_esperada_cm"),
        "look_back": LOOK_BACK,
        "records": records,
    }


@app.get("/sample-alertas", tags=["Inferência"])
def list_alert_samples() -> dict:
    """Lista todas as amostras disponíveis por nível de alerta."""
    manifest_path = SAMPLES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest de amostras não encontrado.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _require_ops() -> OpsStore:
    if ops_store is None:
        raise HTTPException(status_code=503, detail="Store operacional ainda não inicializado.")
    return ops_store


@app.get("/ops/state", tags=["Operação"])
def get_ops_state() -> dict:
    """Snapshot compartilhado: tarefas, previsões e medições de campo do app."""
    return _require_ops().snapshot()


@app.get("/ops/inbox", tags=["Operação"])
def get_ops_inbox(
    equipe: Annotated[str | None, Query(description="Filtra pelo nome da equipe")] = None,
) -> dict:
    """Caixa de entrada do app: comandos abertos enviados pelo dashboard."""
    snap = _require_ops().snapshot()
    dispatches = list(snap.get("dispatches") or [])
    if equipe:
        needle = equipe.strip().lower()
        dispatches = [
            item
            for item in dispatches
            if needle in str(item.get("equipe") or "").lower()
        ]
    abertos = [item for item in dispatches if item.get("status") in OPEN_STATUSES]
    return {
        "version": snap.get("version", 0),
        "equipe": equipe,
        "dispatches": abertos,
    }


@app.post("/ops/presence", tags=["Operação"])
def post_ops_presence(request: PresenceRequest) -> dict:
    """App anuncia o endereço (Flutter em :5050) para o dashboard localizar."""
    return _require_ops().set_presence(request.model_dump())


@app.post("/ops/dispatches", tags=["Operação"], status_code=201)
def create_dispatch(request: DispatchCreateRequest) -> dict:
    """Dashboard envia uma tarefa de poda para o app da equipe."""
    return _require_ops().create_dispatch(request.model_dump())


@app.patch("/ops/dispatches/{dispatch_id}", tags=["Operação"])
def patch_dispatch(dispatch_id: str, request: DispatchPatchRequest) -> dict:
    """App ou dashboard atualiza status, altura medida e comprovação."""
    try:
        return _require_ops().patch_dispatch(
            dispatch_id,
            request.model_dump(exclude_none=True),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/ops/predictions", tags=["Operação"])
def put_predictions(request: PredictionsSyncRequest) -> dict:
    """Dashboard publica o ciclo de previsão para o app consumir os mesmos valores."""
    return _require_ops().set_predictions(request.predictions)


@app.get("/sample/{id_ponto}", tags=["Inferência"])
def get_sample_records(id_ponto: str = "RODO_KM_01") -> dict:
    """Retorna os últimos 20 registros de um ponto do dataset para teste rápido."""
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset não encontrado.")

    df = pd.read_csv(DATASET_PATH)
    point_df = df[df["ID_Ponto"] == id_ponto].sort_values("Data")

    if len(point_df) < LOOK_BACK:
        raise HTTPException(
            status_code=404,
            detail=f"Ponto {id_ponto} não possui registros suficientes.",
        )

    sample = point_df.tail(LOOK_BACK)
    records = sample.to_dict(orient="records")
    return {
        "id_ponto": id_ponto,
        "look_back": LOOK_BACK,
        "records": records,
    }


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Interface não encontrada.")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
