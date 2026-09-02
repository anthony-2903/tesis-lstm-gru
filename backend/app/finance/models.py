from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import tensorflow as tf

from app.phishing.ingestion import sha256_file


FINANCE_MODEL_IDS = ("lstm", "gru", "brnn", "tcn", "transformer")


@tf.keras.utils.register_keras_serializable(package="ThesisFinance")
class NumericPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length: int, width: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.width = width
        self.supports_masking = True
        self.position_embedding = tf.keras.layers.Embedding(sequence_length, width)

    def build(self, input_shape) -> None:
        self.position_embedding.build((self.sequence_length,))
        super().build(input_shape)

    def call(self, values):
        positions = tf.range(start=0, limit=tf.shape(values)[1], delta=1)
        return values + self.position_embedding(positions)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "sequence_length": self.sequence_length, "width": self.width}


@dataclass(frozen=True)
class FinanceModelConfig:
    width: int = 32
    dropout: float = 0.20
    learning_rate: float = 0.001
    transformer_heads: int = 4


class KerasFinanceClassifier:
    def __init__(
        self,
        *,
        model_id: str,
        input_shape: tuple[int, int],
        epochs: int,
        batch_size: int,
        patience: int,
        seed: int,
        config: FinanceModelConfig | None = None,
    ) -> None:
        if model_id not in FINANCE_MODEL_IDS:
            raise ValueError(f"Modelo financiero no soportado: {model_id}")
        self.model_id = model_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.config = config or FinanceModelConfig()
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        self.model = build_finance_keras_model(model_id, input_shape=input_shape, config=self.config)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            ],
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        class_weight: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        callbacks: list[tf.keras.callbacks.Callback] = [tf.keras.callbacks.TerminateOnNaN()]
        if self.patience >= 0 and self.epochs > 1:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_pr_auc",
                    mode="max",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            )
        started = time.perf_counter()
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0,
            shuffle=False,
        )
        return {
            "trainTimeSeconds": float(time.perf_counter() - started),
            "epochsCompleted": int(len(history.history.get("loss", []))),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        }

    def fit_full(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        class_weight: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        """Refit a frozen OOF configuration without consulting validation or test."""
        started = time.perf_counter()
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            class_weight=class_weight,
            verbose=0,
            shuffle=False,
        )
        return {
            "trainTimeSeconds": float(time.perf_counter() - started),
            "epochsCompleted": int(len(history.history.get("loss", []))),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        }

    def predict_proba(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        probabilities = self.model.predict(x, batch_size=self.batch_size, verbose=0).reshape(-1)
        return np.clip(probabilities.astype(np.float64), 0.0, 1.0), float(time.perf_counter() - started)

    def save(self, path: Path) -> tuple[str, int]:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return sha256_file(path)


def build_finance_keras_model(
    model_id: str,
    *,
    input_shape: tuple[int, int],
    config: FinanceModelConfig | None = None,
) -> tf.keras.Model:
    if model_id not in FINANCE_MODEL_IDS:
        raise ValueError(f"Modelo financiero no soportado: {model_id}")
    configuration = config or FinanceModelConfig()
    sequence_length = int(input_shape[0])
    inputs = tf.keras.layers.Input(shape=input_shape, dtype="float32", name="financial_sequence")
    masked = tf.keras.layers.Masking(mask_value=0.0, name="padding_mask")(inputs)

    if model_id == "lstm":
        x = tf.keras.layers.LSTM(configuration.width, name="lstm_encoder")(masked)
    elif model_id == "gru":
        x = tf.keras.layers.GRU(configuration.width, name="gru_encoder")(masked)
    elif model_id == "brnn":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(max(configuration.width // 2, 8)),
            name="bidirectional_gru_encoder",
        )(masked)
    elif model_id == "tcn":
        x = tf.keras.layers.Conv1D(configuration.width, 1, padding="same", name="tcn_projection")(inputs)
        for dilation in (1, 2, 4):
            residual = x
            x = tf.keras.layers.Conv1D(
                configuration.width,
                3,
                padding="causal",
                dilation_rate=dilation,
                activation="relu",
                name=f"tcn_d{dilation}_a",
            )(x)
            x = tf.keras.layers.Dropout(configuration.dropout)(x)
            x = tf.keras.layers.Conv1D(
                configuration.width,
                3,
                padding="causal",
                dilation_rate=dilation,
                name=f"tcn_d{dilation}_b",
            )(x)
            x = tf.keras.layers.Activation("relu")(tf.keras.layers.Add()([residual, x]))
        if sequence_length > 1:
            x = tf.keras.layers.Cropping1D(cropping=(sequence_length - 1, 0), name="current_step")(x)
        x = tf.keras.layers.Flatten()(x)
    else:
        x = tf.keras.layers.Dense(configuration.width, name="numeric_projection")(masked)
        x = NumericPositionEmbedding(sequence_length, configuration.width, name="position_embedding")(x)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=configuration.transformer_heads,
            key_dim=max(configuration.width // configuration.transformer_heads, 4),
            dropout=configuration.dropout,
            name="self_attention",
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attention)
        feed_forward = tf.keras.layers.Dense(configuration.width * 2, activation="relu")(x)
        feed_forward = tf.keras.layers.Dropout(configuration.dropout)(feed_forward)
        feed_forward = tf.keras.layers.Dense(configuration.width)(feed_forward)
        x = tf.keras.layers.LayerNormalization()(x + feed_forward)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dropout(configuration.dropout)(x)
    x = tf.keras.layers.Dense(configuration.width, activation="relu", name="classification_features")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fraud_probability")(x)
    return tf.keras.Model(inputs, outputs, name=f"finance_{model_id}")
