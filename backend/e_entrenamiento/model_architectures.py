"""
Arquitecturas base para entrenar los modelos de la tesis en Google Colab.

Cada constructor recibe la forma de entrada (pasos_temporales, variables)
y devuelve un modelo Keras compilado. La idea es que el frontend solo consuma
metricas/resultados, mientras Colab ejecuta el entrenamiento real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class ModelTrainingConfig:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int


MODEL_TRAINING_CONFIGS: dict[str, ModelTrainingConfig] = {
    "lstm": ModelTrainingConfig(
        name="LSTM",
        epochs=60,
        batch_size=64,
        learning_rate=0.001,
        patience=8,
    ),
    "gru": ModelTrainingConfig(
        name="GRU",
        epochs=50,
        batch_size=64,
        learning_rate=0.001,
        patience=7,
    ),
    "transformer": ModelTrainingConfig(
        name="Transformer",
        epochs=80,
        batch_size=32,
        learning_rate=0.0005,
        patience=10,
    ),
    "tcn": ModelTrainingConfig(
        name="TCN",
        epochs=45,
        batch_size=64,
        learning_rate=0.001,
        patience=7,
    ),
}


def _compile(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_lstm(input_shape: tuple[int, int], learning_rate: float = 0.001) -> keras.Model:
    """LSTM: mejor baseline recurrente para dependencias temporales largas."""
    inputs = keras.Input(shape=input_shape, name="sequence_input")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.0)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="anomaly_probability")(x)
    model = keras.Model(inputs, outputs, name="lstm_anomaly_detector")
    return _compile(model, learning_rate)


def build_gru(input_shape: tuple[int, int], learning_rate: float = 0.001) -> keras.Model:
    """GRU: variante recurrente mas ligera y rapida que LSTM."""
    inputs = keras.Input(shape=input_shape, name="sequence_input")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.GRU(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.0)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(48, return_sequences=False, dropout=0.2, recurrent_dropout=0.0)(x)
    x = layers.Dense(48, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="anomaly_probability")(x)
    model = keras.Model(inputs, outputs, name="gru_anomaly_detector")
    return _compile(model, learning_rate)


def _transformer_encoder(
    inputs: tf.Tensor,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
) -> tf.Tensor:
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    y = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    y = layers.Dropout(dropout)(y)
    y = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(y)
    return layers.LayerNormalization(epsilon=1e-6)(x + y)


def build_transformer(input_shape: tuple[int, int], learning_rate: float = 0.0005) -> keras.Model:
    """Transformer: captura dependencias globales mediante auto-atencion."""
    inputs = keras.Input(shape=input_shape, name="sequence_input")
    x = layers.Dense(64)(inputs)
    x = _transformer_encoder(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.2)
    x = _transformer_encoder(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="anomaly_probability")(x)
    model = keras.Model(inputs, outputs, name="transformer_anomaly_detector")
    return _compile(model, learning_rate)


def _tcn_block(inputs: tf.Tensor, filters: int, dilation_rate: int, dropout: float) -> tf.Tensor:
    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)

    residual = inputs
    if inputs.shape[-1] != filters:
        residual = layers.Conv1D(filters=filters, kernel_size=1, padding="same")(inputs)
    return layers.Activation("relu")(x + residual)


def build_tcn(input_shape: tuple[int, int], learning_rate: float = 0.001) -> keras.Model:
    """TCN: convoluciones causales dilatadas para patrones temporales eficientes."""
    inputs = keras.Input(shape=input_shape, name="sequence_input")
    x = inputs
    for dilation in (1, 2, 4, 8):
        x = _tcn_block(x, filters=64, dilation_rate=dilation, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="anomaly_probability")(x)
    model = keras.Model(inputs, outputs, name="tcn_anomaly_detector")
    return _compile(model, learning_rate)


MODEL_BUILDERS: dict[str, Callable[[tuple[int, int], float], keras.Model]] = {
    "lstm": build_lstm,
    "gru": build_gru,
    "transformer": build_transformer,
    "tcn": build_tcn,
}


def build_model(model_key: str, input_shape: tuple[int, int]) -> keras.Model:
    key = model_key.lower()
    if key not in MODEL_BUILDERS:
        valid = ", ".join(MODEL_BUILDERS)
        raise ValueError(f"Modelo no soportado: {model_key}. Opciones validas: {valid}")

    config = MODEL_TRAINING_CONFIGS[key]
    return MODEL_BUILDERS[key](input_shape, config.learning_rate)
