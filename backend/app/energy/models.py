from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import tensorflow as tf


THESIS_MODEL_IDS = ("lstm", "gru", "brnn", "tcn", "transformer")


@tf.keras.utils.register_keras_serializable(package="ThesisEnergy")
class LearnablePositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length: int, model_width: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.model_width = model_width
        self.embedding = tf.keras.layers.Embedding(sequence_length, model_width)

    def build(self, input_shape) -> None:
        self.embedding.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        position_values = self.embedding(positions)
        return inputs + tf.expand_dims(position_values, axis=0)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "sequence_length": self.sequence_length,
            "model_width": self.model_width,
        }


class EnergyRegressor(ABC):
    model_id: str

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> Path | None:
        raise NotImplementedError


class NaivePersistenceRegressor(EnergyRegressor):
    model_id = "naive_persistence"

    def __init__(self, target_feature_index: int) -> None:
        self.target_feature_index = target_feature_index

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray) -> dict[str, Any]:
        return {"train_time_seconds": 0.0, "epochs_completed": 0, "history": {}}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x[:, -1, self.target_feature_index], dtype=np.float32)

    def save(self, path: Path) -> Path | None:
        return None


class KerasEnergyRegressor(EnergyRegressor):
    def __init__(self, *, input_shape: tuple[int, int], learning_rate: float = 0.001, epochs: int = 20, batch_size: int = 32, patience: int = 4, seed: int = 42) -> None:
        self.tf = tf
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        self.model = self._build_model(tf, input_shape)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    @abstractmethod
    def _build_model(self, tf, input_shape: tuple[int, int]):
        raise NotImplementedError

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray) -> dict[str, Any]:
        callback = self.tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        started = time.perf_counter()
        history = self.model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=self.epochs, batch_size=self.batch_size, callbacks=[callback], verbose=0, shuffle=False)
        return {
            "train_time_seconds": float(time.perf_counter() - started),
            "epochs_completed": int(len(history.history.get("loss", []))),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x, batch_size=self.batch_size, verbose=0).reshape(-1).astype(np.float32)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return path


class KerasLSTMRegressor(KerasEnergyRegressor):
    model_id = "lstm"

    def _build_model(self, tf, input_shape: tuple[int, int]):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape, name="energy_sequence"),
            tf.keras.layers.LSTM(32, name="lstm_encoder"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, name="forecast"),
        ], name="energy_lstm")


class KerasGRURegressor(KerasEnergyRegressor):
    model_id = "gru"

    def _build_model(self, tf, input_shape: tuple[int, int]):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape, name="energy_sequence"),
            tf.keras.layers.GRU(32, name="gru_encoder"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, name="forecast"),
        ], name="energy_gru")


class KerasBRNNRegressor(KerasEnergyRegressor):
    model_id = "brnn"

    def _build_model(self, tf, input_shape: tuple[int, int]):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape, name="energy_sequence"),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24), name="bidirectional_lstm_encoder"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, name="forecast"),
        ], name="energy_brnn")


class KerasTCNRegressor(KerasEnergyRegressor):
    model_id = "tcn"

    def _build_model(self, tf, input_shape: tuple[int, int]):
        inputs = tf.keras.layers.Input(shape=input_shape, name="energy_sequence")
        x = tf.keras.layers.Conv1D(32, 1, padding="same", name="tcn_input_projection")(inputs)
        for dilation in (1, 2, 4, 8):
            residual = x
            x = tf.keras.layers.Conv1D(32, 3, padding="causal", dilation_rate=dilation, activation="relu", name=f"tcn_causal_d{dilation}_a")(x)
            x = tf.keras.layers.Dropout(0.10)(x)
            x = tf.keras.layers.Conv1D(32, 3, padding="causal", dilation_rate=dilation, name=f"tcn_causal_d{dilation}_b")(x)
            x = tf.keras.layers.Add(name=f"tcn_residual_d{dilation}")([residual, x])
            x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, name="forecast")(x)
        return tf.keras.Model(inputs, outputs, name="energy_tcn")


class KerasTransformerRegressor(KerasEnergyRegressor):
    model_id = "transformer"

    def _build_model(self, tf, input_shape: tuple[int, int]):
        sequence_length = input_shape[0]
        model_width = 32
        inputs = tf.keras.layers.Input(shape=input_shape, name="energy_sequence")
        x = tf.keras.layers.Dense(model_width, name="transformer_projection")(inputs)
        x = LearnablePositionEmbedding(sequence_length, model_width, name="position_embedding")(x)
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.10, name="self_attention")(x, x, use_causal_mask=True)
        x = tf.keras.layers.LayerNormalization(name="attention_norm")(x + attention)
        feed_forward = tf.keras.layers.Dense(64, activation="relu", name="ffn_expand")(x)
        feed_forward = tf.keras.layers.Dropout(0.10)(feed_forward)
        feed_forward = tf.keras.layers.Dense(model_width, name="ffn_project")(feed_forward)
        x = tf.keras.layers.LayerNormalization(name="ffn_norm")(x + feed_forward)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, name="forecast")(x)
        return tf.keras.Model(inputs, outputs, name="energy_transformer")


MODEL_BUILDERS: dict[str, Callable[..., KerasEnergyRegressor]] = {
    "lstm": KerasLSTMRegressor,
    "gru": KerasGRURegressor,
    "brnn": KerasBRNNRegressor,
    "tcn": KerasTCNRegressor,
    "transformer": KerasTransformerRegressor,
}


def build_energy_models(*, input_shape: tuple[int, int], target_feature_index: int, model_ids: tuple[str, ...] = THESIS_MODEL_IDS, epochs: int = 20, batch_size: int = 32, seed: int = 42, include_naive: bool = True) -> list[EnergyRegressor]:
    unknown = sorted(set(model_ids) - set(MODEL_BUILDERS))
    if unknown:
        raise ValueError(f"Modelos energeticos no soportados: {', '.join(unknown)}")
    models: list[EnergyRegressor] = [NaivePersistenceRegressor(target_feature_index)] if include_naive else []
    for model_id in model_ids:
        models.append(MODEL_BUILDERS[model_id](input_shape=input_shape, epochs=epochs, batch_size=batch_size, seed=seed))
    return models
