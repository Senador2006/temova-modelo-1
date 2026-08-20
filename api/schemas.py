from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class DailyRecord(BaseModel):
    id_ponto: str = Field(..., alias="ID_Ponto", examples=["RODO_KM_01"])
    data: date = Field(..., alias="Data", examples=["2023-01-21"])
    temperatura_media_c: float = Field(..., alias="Temperatura_Media_C", examples=[24.5])
    precipitacao_diaria_mm: float = Field(..., alias="Precipitacao_Diaria_mm", examples=[3.2])
    umidade_relativa_pct: float = Field(..., alias="Umidade_Relativa_Pct", examples=[62.0])
    radiacao_solar_mjm2: float = Field(..., alias="Radiacao_Solar_MJm2", examples=[18.5])
    evapotranspiracao_mm: float = Field(..., alias="Evapotranspiracao_mm", examples=[4.1])
    qualidade_ar_iqar: int = Field(..., alias="Qualidade_Ar_IQAr", examples=[45])
    emissao_co2_gm2: float = Field(..., alias="Emissao_CO2_gm2", examples=[430.0])
    ndvi: float = Field(..., alias="NDVI", examples=[0.42])
    evi: float = Field(..., alias="EVI", examples=[0.28])
    umidade_solo_pct: float = Field(..., alias="Umidade_Solo_Pct", examples=[15.0])
    crescimento_semanal_cm: float = Field(..., alias="Crescimento_Semanal_cm", examples=[3.5])
    crescimento_mensal_cm: float = Field(..., alias="Crescimento_Mensal_cm", examples=[14.0])
    especie_predominante: str = Field(..., alias="Especie_Predominante", examples=["Urochloa brizantha"])
    tipo_solo: str = Field(..., alias="Tipo_Solo", examples=["Latossolo Vermelho"])
    declividade_pct: float = Field(..., alias="Declividade_Pct", examples=[8.5])
    altitude_m: float = Field(..., alias="Altitude_m", examples=[820.0])
    orientacao_encosta: str = Field(..., alias="Orientacao_Encosta", examples=["Sul"])
    data_ultima_poda: date = Field(..., alias="Data_Ultima_Poda", examples=["2022-10-21"])
    dias_desde_poda: int = Field(..., alias="Dias_Desde_Poda", examples=[92])
    intensidade_ultima_poda: str = Field(..., alias="Intensidade_Ultima_Poda", examples=["Moderada"])
    dia_ano: int = Field(..., alias="Dia_Ano", examples=[21])
    semana_ano: int = Field(..., alias="Semana_Ano", examples=[3])
    mes: int = Field(..., alias="Mes", examples=[1])
    estacao: str = Field(..., alias="Estacao", examples=["Verão"])
    chuva_acum_7d_mm: float = Field(..., alias="Chuva_Acum_7d_mm", examples=[12.5])
    chuva_acum_15d_mm: float = Field(..., alias="Chuva_Acum_15d_mm", examples=[25.0])
    chuva_acum_30d_mm: float = Field(..., alias="Chuva_Acum_30d_mm", examples=[48.0])
    temp_media_7d_c: float = Field(..., alias="Temp_Media_7d_C", examples=[24.0])
    temp_media_15d_c: float = Field(..., alias="Temp_Media_15d_C", examples=[23.5])
    temp_media_30d_c: float = Field(..., alias="Temp_Media_30d_C", examples=[23.0])
    dias_sem_chuva: int = Field(..., alias="Dias_Sem_Chuva", examples=[2])
    crescimento_acumulado_cm: Optional[float] = Field(
        default=None,
        alias="Crescimento_Acumulado_cm",
        description="Opcional na inferência; ignorado pelo modelo.",
    )

    model_config = {"populate_by_name": True}


class PredictionRequest(BaseModel):
    records: list[DailyRecord] = Field(
        ...,
        min_length=20,
        description="Histórico diário consecutivo (mínimo 20 dias, conforme look_back do modelo).",
    )


class PredictionResponse(BaseModel):
    prediction_id: str
    crescimento_acumulado_cm: float
    limite_alerta_cm: float = 9.0
    limite_poda_cm: float = 10.0
    acima_limite_poda: bool
    alerta_preventivo: bool
    recomendacao_poda: str
    nivel_alerta: str
    nivel_label: str
    mensagem_alerta: str
    distancia_limite_cm: float
    distancia_alerta_cm: float
    proximo_limite: bool
    model_version: str
    look_back: int
    registros_utilizados: int
    id_ponto: str
    data_referencia: str
    latency_ms: float
    unidade: str = "cm"


class PredictionLogEntry(BaseModel):
    prediction_id: str
    timestamp: str
    id_ponto: str
    data_referencia: str
    crescimento_acumulado_cm: float
    nivel_alerta: str
    nivel_label: str
    alerta_preventivo: bool
    acima_limite_poda: bool
    distancia_limite_cm: float
    distancia_alerta_cm: float
    model_version: str
    latency_ms: float
    source_system: str | None = None


class PredictionLogsResponse(BaseModel):
    total: int
    limit: int
    logs: list[PredictionLogEntry]


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    window_hours: int
    predictions_in_window: int
    avg_crescimento_cm: float | None
    alerta_preventivo_count: int
    poda_necessaria_count: int
    alerta_preventivo_rate_pct: float
    poda_necessaria_rate_pct: float
    by_nivel_alerta_all_time: dict[str, int]
    by_nivel_alerta_window: dict[str, int]
    latest_by_ponto: list[dict]
    ultima_predicao: dict | None = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    look_back: int


class DispatchCreateRequest(BaseModel):
    idPonto: str
    equipe: str
    situacao: str = ""
    nivelAlerta: str = "amarelo"
    crescimentoCm: float = 0
    limitePodaCm: float = 10
    limiteAlertaCm: float = 9
    road: str = ""
    km: float = 0
    local: str = ""
    sentido: str = ""
    titulo: str = ""


class DispatchPatchRequest(BaseModel):
    status: str | None = None
    alturaFinal: float | None = None
    foto: str | None = None
    source: str = "app"


class PredictionsSyncRequest(BaseModel):
    predictions: dict[str, dict]


class PresenceRequest(BaseModel):
    url: str = "http://127.0.0.1:5050"
    equipe: str = "Equipe Delta"
    client: str = "app"
