import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def _resolve_dataset_path() -> Path:
    if env_path := os.getenv("DATASET_PATH"):
        return Path(env_path)

    local_data = BASE_DIR / "data" / "dataset_rodoanel_10k.csv"
    if local_data.exists():
        return local_data

    return BASE_DIR.parent / "dataset_rodoanel_10k.csv"


DATASET_PATH = _resolve_dataset_path()

TARGET_VARIABLE = "Crescimento_Acumulado_cm"
LOOK_BACK = 20
MODEL_VERSION = "1.2.0"

LIMITE_ALERTA_CM = 9.0
LIMITE_PODA_CM = 10.0
OVERSAMPLE_THRESHOLD_CM = 10.0
OVERSAMPLE_FACTOR = 3

LOG_DIR = BASE_DIR / "logs"
PREDICTION_LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", str(LOG_DIR / "predictions.jsonl")))

CATEGORICAL_COLS = [
    "ID_Ponto",
    "Especie_Predominante",
    "Tipo_Solo",
    "Orientacao_Encosta",
    "Intensidade_Ultima_Poda",
    "Estacao",
]

RAW_COLUMNS = [
    "ID_Ponto",
    "Data",
    "Temperatura_Media_C",
    "Precipitacao_Diaria_mm",
    "Umidade_Relativa_Pct",
    "Radiacao_Solar_MJm2",
    "Evapotranspiracao_mm",
    "Qualidade_Ar_IQAr",
    "Emissao_CO2_gm2",
    "NDVI",
    "EVI",
    "Umidade_Solo_Pct",
    "Crescimento_Acumulado_cm",
    "Crescimento_Semanal_cm",
    "Crescimento_Mensal_cm",
    "Especie_Predominante",
    "Tipo_Solo",
    "Declividade_Pct",
    "Altitude_m",
    "Orientacao_Encosta",
    "Data_Ultima_Poda",
    "Dias_Desde_Poda",
    "Intensidade_Ultima_Poda",
    "Dia_Ano",
    "Semana_Ano",
    "Mes",
    "Estacao",
    "Chuva_Acum_7d_mm",
    "Chuva_Acum_15d_mm",
    "Chuva_Acum_30d_mm",
    "Temp_Media_7d_C",
    "Temp_Media_15d_C",
    "Temp_Media_30d_C",
    "Dias_Sem_Chuva",
]
