import numpy as np
import tensorflow as tf

WEIGHTS = "saved_models/isl_lstm_class_weighted_legacy_weights.npz"
OUTPUT = "saved_models/isl_lstm_class_weighted_legacy_compatible.h5"


def build_model():
    inputs = tf.keras.Input(
        shape=(60, 258),
        name="Input"
    )

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True),
        name="BiLSTM_1"
    )(inputs)

    x = tf.keras.layers.Dropout(
        0.30,
        name="Dropout_1"
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64),
        name="BiLSTM_2"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="Dropout_2"
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="Dense_1"
    )(x)

    x = tf.keras.layers.Dropout(
        0.20,
        name="Dropout_3"
    )(x)

    outputs = tf.keras.layers.Dense(
        24,
        activation="softmax",
        name="Output"
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="ISL_BiLSTM"
    )


model = build_model()

print("Model created")
print("Input :", model.input_shape)
print("Output:", model.output_shape)
print("Params:", model.count_params())

data = np.load(WEIGHTS)
weights = [data[key] for key in data.files]

print("Imported weight arrays:", len(weights))

model_weights = model.get_weights()

if len(weights) != len(model_weights):
    raise RuntimeError(
        f"Weight count mismatch: source={len(weights)}, "
        f"model={len(model_weights)}"
    )

for i, (src, dst) in enumerate(zip(weights, model_weights)):
    if src.shape != dst.shape:
        raise RuntimeError(
            f"Shape mismatch at weight {i}: "
            f"source={src.shape}, model={dst.shape}"
        )

model.set_weights(weights)

print("Weights loaded successfully")

model.save(OUTPUT)

print("Compatible model saved:", OUTPUT)