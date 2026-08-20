import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_COLS,
    LOOK_BACK,
    LIMITE_ALERTA_CM,
    LIMITE_PODA_CM,
    TARGET_VARIABLE,
)


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica o mesmo pipeline de engenharia de features do notebook."""
    processed = df.copy()
    processed["Data"] = pd.to_datetime(processed["Data"])
    processed["Data_Ultima_Poda"] = pd.to_datetime(processed["Data_Ultima_Poda"])
    processed = processed.sort_values(by=["ID_Ponto", "Data"]).reset_index(drop=True)

    processed = pd.get_dummies(processed, columns=CATEGORICAL_COLS, drop_first=True)
    processed = processed.drop("Data_Ultima_Poda", axis=1)
    processed["Data"] = processed["Data"].astype("int64") / 1e9

    features = processed.drop(columns=[TARGET_VARIABLE])
    target = processed[TARGET_VARIABLE]
    return features, target


def align_features(features: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Garante que as colunas coincidam com as usadas no treinamento."""
    aligned = features.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = 0

    extra_columns = set(aligned.columns) - set(expected_columns)
    if extra_columns:
        aligned = aligned.drop(columns=list(extra_columns))

    return aligned[expected_columns]


def create_sequences(
    scaled_features: pd.DataFrame,
    id_series: pd.Series,
    look_back: int = LOOK_BACK,
) -> np.ndarray:
    """Cria sequências temporais agrupadas por ID_Ponto."""
    sequences = []
    grouped = scaled_features.copy()
    grouped["__id__"] = id_series.values

    for _, group in grouped.groupby("__id__", sort=False):
        feature_rows = group.drop(columns=["__id__"])
        if len(feature_rows) < look_back:
            continue
        values = feature_rows.values
        sequences.append(values[-look_back:])

    if not sequences:
        raise ValueError(
            f"São necessários pelo menos {look_back} registros consecutivos por ponto."
        )

    return np.array(sequences)


def fit_preprocessors(df: pd.DataFrame) -> dict:
    """Treina scalers e retorna artefatos de pré-processamento."""
    features, target = preprocess_dataframe(df)

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    scaled_features = scaler_features.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)

    scaler_target.fit(target.to_frame())

    return {
        "feature_columns": features.columns.tolist(),
        "scaler_features": scaler_features,
        "scaler_target": scaler_target,
    }


def transform_for_prediction(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler_features: MinMaxScaler,
    look_back: int = LOOK_BACK,
) -> np.ndarray:
    """Transforma registros brutos em sequência pronta para inferência."""
    if TARGET_VARIABLE not in df.columns:
        df = df.copy()
        df[TARGET_VARIABLE] = 0.0

    features, _ = preprocess_dataframe(df)
    features = align_features(features, feature_columns)
    scaled = scaler_features.transform(features)
    scaled_df = pd.DataFrame(scaled, columns=feature_columns, index=df.index)

    return create_sequences(scaled_df, df["ID_Ponto"], look_back=look_back)


def inverse_transform_target(values: np.ndarray, scaler_target: MinMaxScaler) -> np.ndarray:
    reshaped = values.reshape(-1, 1)
    return scaler_target.inverse_transform(reshaped).flatten()


def save_artifacts(artifacts: dict, output_dir: Path = ARTIFACTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts["scaler_features"], output_dir / "scaler_features.joblib")
    joblib.dump(artifacts["scaler_target"], output_dir / "scaler_target.joblib")

    metadata = {
        "feature_columns": artifacts["feature_columns"],
        "target_variable": TARGET_VARIABLE,
        "look_back": LOOK_BACK,
        "limite_alerta_cm": LIMITE_ALERTA_CM,
        "limite_poda_cm": LIMITE_PODA_CM,
        "model_version": artifacts.get("model_version", "1.2.0"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    metadata = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        **metadata,
        "scaler_features": joblib.load(artifacts_dir / "scaler_features.joblib"),
        "scaler_target": joblib.load(artifacts_dir / "scaler_target.joblib"),
    }
