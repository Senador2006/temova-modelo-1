import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "torch")

from keras.models import load_model

from ml.alert_levels import calcular_nivel_alerta
from ml.config import ARTIFACTS_DIR, LOOK_BACK, LIMITE_ALERTA_CM, LIMITE_PODA_CM
from ml.preprocessing import inverse_transform_target, load_artifacts, transform_for_prediction


class GrassGrowthPredictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts = load_artifacts(artifacts_dir)
        self.model = load_model(artifacts_dir / "lstm_model.keras")
        self.look_back = self.artifacts.get("look_back", LOOK_BACK)

    @property
    def model_version(self) -> str:
        return self.artifacts.get("model_version", "unknown")

    def predict(self, records: list[dict]) -> dict:
        if len(records) < self.look_back:
            raise ValueError(
                f"São necessários pelo menos {self.look_back} registros diários consecutivos."
            )

        df = pd.DataFrame(records)
        sequence = transform_for_prediction(
            df,
            self.artifacts["feature_columns"],
            self.artifacts["scaler_features"],
            look_back=self.look_back,
        )

        scaled_prediction = self.model.predict(sequence[-1:], verbose=0)
        growth_cm = float(
            inverse_transform_target(
                scaled_prediction.flatten(),
                self.artifacts["scaler_target"],
            )[0]
        )
        growth_cm = round(growth_cm, 2)
        limite_poda = self.artifacts.get("limite_poda_cm", LIMITE_PODA_CM)
        limite_alerta = self.artifacts.get("limite_alerta_cm", LIMITE_ALERTA_CM)
        alerta = calcular_nivel_alerta(growth_cm, limite_poda, limite_alerta)

        return {
            "crescimento_acumulado_cm": growth_cm,
            "limite_alerta_cm": limite_alerta,
            "limite_poda_cm": limite_poda,
            "acima_limite_poda": growth_cm >= limite_poda,
            "alerta_preventivo": growth_cm >= limite_alerta,
            "recomendacao_poda": alerta["mensagem_alerta"],
            **alerta,
            "model_version": self.model_version,
            "look_back": self.look_back,
            "registros_utilizados": len(records),
        }
