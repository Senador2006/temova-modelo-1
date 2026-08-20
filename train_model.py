"""
Treina o modelo LSTM do Protótipo 1 e exporta artefatos para a API MLOps.

Uso:
    python train_model.py
    python train_model.py --epochs 50
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "torch")

from ml.config import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    LOOK_BACK,
    MODEL_VERSION,
    OVERSAMPLE_FACTOR,
    OVERSAMPLE_THRESHOLD_CM,
    TARGET_VARIABLE,
)
from ml.model_builder import build_lstm_model
from ml.preprocessing import fit_preprocessors, inverse_transform_target, preprocess_dataframe, save_artifacts


def create_training_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler_features,
    scaler_target,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray]:
    features, target = preprocess_dataframe(df)

    scaled_features = scaler_features.transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=feature_columns, index=df.index)
    scaled_target = scaler_target.transform(target.to_frame())

    full_scaled_df = scaled_features_df.copy()
    full_scaled_df[TARGET_VARIABLE] = scaled_target.flatten()
    full_scaled_df["ID_Ponto"] = df["ID_Ponto"].values

    xs, ys = [], []
    for _, group in full_scaled_df.groupby("ID_Ponto", sort=False):
        feature_rows = group.drop(columns=["ID_Ponto", TARGET_VARIABLE])
        target_rows = group[TARGET_VARIABLE]

        if len(feature_rows) <= look_back:
            continue

        values = feature_rows.values
        targets = target_rows.values
        for index in range(len(values) - look_back):
            xs.append(values[index : index + look_back])
            ys.append(targets[index + look_back])

    return np.array(xs), np.array(ys).reshape(-1, 1)


def oversample_above_limit(
    x: np.ndarray,
    y: np.ndarray,
    scaler_target,
    threshold_cm: float = OVERSAMPLE_THRESHOLD_CM,
    factor: int = OVERSAMPLE_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Duplica sequências cujo alvo está acima do limite de poda (faixa minoritária)."""
    y_original = inverse_transform_target(y.flatten(), scaler_target)
    above_idx = np.where(y_original >= threshold_cm)[0]

    if len(above_idx) == 0 or factor <= 1:
        return x, y

    repeats = factor - 1
    x_extra = np.repeat(x[above_idx], repeats, axis=0)
    y_extra = np.repeat(y[above_idx], repeats, axis=0)

    print(
        f"Oversampling: {len(above_idx)} sequencias >= {threshold_cm} cm "
        f"-> +{len(x_extra)} amostras (fator {factor}x)"
    )

    return np.concatenate([x, x_extra], axis=0), np.concatenate([y, y_extra], axis=0)


def train(epochs: int = 50, batch_size: int = 32) -> None:
    print(f"Carregando dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    print("Preparando pré-processadores...")
    preprocessors = fit_preprocessors(df)
    preprocessors["model_version"] = MODEL_VERSION

    print("Criando sequências temporais...")
    x, y = create_training_sequences(
        df,
        preprocessors["feature_columns"],
        preprocessors["scaler_features"],
        preprocessors["scaler_target"],
        LOOK_BACK,
    )

    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    x_train, y_train = oversample_above_limit(
        x_train,
        y_train,
        preprocessors["scaler_target"],
        threshold_cm=OVERSAMPLE_THRESHOLD_CM,
        factor=OVERSAMPLE_FACTOR,
    )

    num_features = x_train.shape[2]
    print(f"Treinando LSTM ({epochs} epochs) | X_train={x_train.shape} | features={num_features}")

    model = build_lstm_model(LOOK_BACK, num_features)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )

    predictions = model.predict(x_test, verbose=0)
    y_true = inverse_transform_target(y_test.flatten(), preprocessors["scaler_target"])
    y_pred = inverse_transform_target(predictions.flatten(), preprocessors["scaler_target"])
    mae = np.mean(np.abs(y_true - y_pred))
    mask_acima = y_true >= OVERSAMPLE_THRESHOLD_CM
    if mask_acima.any():
        mae_acima = np.mean(np.abs(y_true[mask_acima] - y_pred[mask_acima]))
        print(f"MAE faixa >= {OVERSAMPLE_THRESHOLD_CM} cm: {mae_acima:.4f} cm")
    print(f"MAE no conjunto de teste: {mae:.4f} cm")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "lstm_model.keras"
    model.save(model_path)
    save_artifacts(preprocessors, ARTIFACTS_DIR)

    print(f"Modelo salvo em: {model_path}")
    print(f"Artefatos salvos em: {ARTIFACTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo LSTM de crescimento da grama")
    parser.add_argument("--epochs", type=int, default=50, help="Número de epochs (padrão: 50)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
