from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


class Trainer:

    def __init__(self, config):

        self.config = config

    def get_callbacks(self):

        return [

            EarlyStopping(

                monitor="val_loss",

                mode="min",

                patience=10,

                restore_best_weights=True,

                verbose=1,

            ),

            ModelCheckpoint(

                filepath=self.config.BEST_MODEL,

                monitor="val_loss",

                mode="min",

                save_best_only=True,

                save_weights_only=False,

                verbose=1,

            ),

            ReduceLROnPlateau(

                monitor="val_loss",

                factor=0.5,

                patience=5,

                verbose=1,

            ),

            TensorBoard(

                log_dir=self.config.TENSORBOARD_LOGS,

            ),

        ]

    def train(

        self,

        model,

        X_train,

        y_train,

        X_val,

        y_val,

    ):

        history = model.fit(

            X_train,

            y_train,

            validation_data=(X_val, y_val),

            epochs=self.config.EPOCHS,

            batch_size=self.config.BATCH_SIZE,

            callbacks=self.get_callbacks(),

            verbose=1,

        )

        return history