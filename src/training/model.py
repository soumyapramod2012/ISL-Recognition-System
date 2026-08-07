import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
)

from src.training.config import (
    SEQUENCE_LENGTH,
    LANDMARK_SIZE,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DENSE_UNITS,
    DROPOUT_1,
    DROPOUT_2,
    DROPOUT_3,
    LEARNING_RATE,
)


class ISLModel:

    def __init__(self):
        pass

    def build(self, num_classes):

        inputs = Input(
            shape=(SEQUENCE_LENGTH, LANDMARK_SIZE),
            name="Input"
        )

        x = Bidirectional(
                LSTM(
                    units=LSTM_UNITS_1,
                    return_sequences=True,
                ),
                name="BiLSTM_1",
        )(inputs)

        x = Dropout(
            rate=DROPOUT_1,
            name="Dropout_1"
        )(x)

        x = Bidirectional(
                LSTM(
                    units=LSTM_UNITS_2,
                    return_sequences=False,
                ),
                name="BiLSTM_2",
        )(x)

        x = Dropout(
            rate=DROPOUT_2,
            name="Dropout_2"
        )(x)

        x = Dense(
            units=DENSE_UNITS,
            activation="relu",
            name="Dense_1"
        )(x)

        x = Dropout(
            rate=DROPOUT_3,
            name="Dropout_3"
        )(x)

        outputs = Dense(
            units=num_classes,
            activation="softmax",
            name="Output"
        )(x)

        model = Model(
            inputs=inputs,
            outputs=outputs,
            name="ISL_BiLSTM"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            ),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy"
            ]
        )

        return model