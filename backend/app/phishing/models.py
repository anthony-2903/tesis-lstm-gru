from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import tensorflow as tf


PHISHING_MODEL_IDS = ("lstm", "gru", "brnn", "tcn", "transformer")


@tf.keras.utils.register_keras_serializable(package="ThesisPhishing")
class TokenPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length: int, vocabulary_size: int, embedding_dimension: int, use_mask: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.use_mask = use_mask
        self.supports_masking = use_mask
        self.token_embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dimension)
        self.position_embedding = tf.keras.layers.Embedding(sequence_length, embedding_dimension)

    def build(self, input_shape) -> None:
        self.token_embedding.build(input_shape)
        self.position_embedding.build((self.sequence_length,))
        super().build(input_shape)

    def call(self, token_ids):
        positions = tf.range(start=0, limit=tf.shape(token_ids)[1], delta=1)
        embedded = self.token_embedding(token_ids) + self.position_embedding(positions)
        mask = tf.cast(tf.not_equal(token_ids, 0), embedded.dtype)
        return embedded * tf.expand_dims(mask, axis=-1)

    def compute_mask(self, token_ids, mask=None):
        return tf.keras.ops.not_equal(token_ids, 0) if self.use_mask else None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "sequence_length": self.sequence_length,
            "vocabulary_size": self.vocabulary_size,
            "embedding_dimension": self.embedding_dimension,
            "use_mask": self.use_mask,
        }


@dataclass(frozen=True)
class PhishingModelConfig:
    embedding_dimension: int = 32
    width: int = 32
    dropout: float = 0.20
    learning_rate: float = 0.001


class KerasPhishingClassifier:
    def __init__(
        self,
        *,
        model_id: str,
        vocabulary_size: int,
        sequence_length: int,
        epochs: int,
        batch_size: int,
        patience: int,
        seed: int,
        config: PhishingModelConfig | None = None,
    ) -> None:
        if model_id not in PHISHING_MODEL_IDS:
            raise ValueError(f"Modelo de phishing no soportado: {model_id}")
        self.model_id = model_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.config = config or PhishingModelConfig()
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        self.model = build_phishing_keras_model(
            model_id,
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            config=self.config,
        )
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
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=self.patience,
                restore_best_weights=True,
            ))
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
            shuffle=True,
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
        """Refit with a frozen epoch count after model selection has finished."""
        started = time.perf_counter()
        history = self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            class_weight=class_weight,
            verbose=0,
            shuffle=True,
        )
        return {
            "trainTimeSeconds": float(time.perf_counter() - started),
            "epochsCompleted": int(len(history.history.get("loss", []))),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
            "fitPolicy": "full_development_refit_with_oof_selected_epochs",
        }

    def predict_proba(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        probabilities = self.model.predict(x, batch_size=self.batch_size, verbose=0).reshape(-1)
        elapsed = time.perf_counter() - started
        return np.clip(probabilities.astype(np.float64), 0.0, 1.0), elapsed

    def save(self, path: Path) -> tuple[str, int]:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return hash_keras_artifact(path)


def build_phishing_keras_model(
    model_id: str,
    *,
    vocabulary_size: int,
    sequence_length: int,
    config: PhishingModelConfig | None = None,
) -> tf.keras.Model:
    configuration = config or PhishingModelConfig()
    inputs = tf.keras.layers.Input(shape=(sequence_length,), dtype="int32", name="character_tokens")
    x = TokenPositionEmbedding(
        sequence_length,
        vocabulary_size,
        configuration.embedding_dimension,
        use_mask=model_id != "tcn",
        name="token_position_embedding",
    )(inputs)
    if model_id == "lstm":
        x = tf.keras.layers.LSTM(configuration.width, name="lstm_encoder")(x)
    elif model_id == "gru":
        x = tf.keras.layers.GRU(configuration.width, name="gru_encoder")(x)
    elif model_id == "brnn":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(max(configuration.width // 2, 8)),
            name="bidirectional_gru_encoder",
        )(x)
    elif model_id == "tcn":
        x = tf.keras.layers.Conv1D(configuration.width, 1, padding="same", name="tcn_projection")(x)
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
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    elif model_id == "transformer":
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=max(configuration.embedding_dimension // 4, 4),
            dropout=configuration.dropout,
            name="self_attention",
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attention)
        feed_forward = tf.keras.layers.Dense(configuration.width * 2, activation="relu")(x)
        feed_forward = tf.keras.layers.Dropout(configuration.dropout)(feed_forward)
        feed_forward = tf.keras.layers.Dense(configuration.embedding_dimension)(feed_forward)
        x = tf.keras.layers.LayerNormalization()(x + feed_forward)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(f"Modelo de phishing no soportado: {model_id}")
    x = tf.keras.layers.Dropout(configuration.dropout)(x)
    x = tf.keras.layers.Dense(configuration.width, activation="relu", name="classification_features")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="phishing_probability")(x)
    return tf.keras.Model(inputs, outputs, name=f"phishing_{model_id}")


def hash_keras_artifact(path: Path) -> tuple[str, int]:
    """Hash a Keras file or SavedModel directory with one stable contract.

    Keras 3 normally writes a single ``.keras`` container.  Including the
    relative file name keeps the same contract when a SavedModel directory is
    used, and avoids mixing this digest with a plain byte-only file hash.
    """
    import hashlib

    digest = hashlib.sha256()
    total = 0
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in files:
        relative = item.relative_to(path).as_posix() if path.is_dir() else item.name
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
    return digest.hexdigest(), total
