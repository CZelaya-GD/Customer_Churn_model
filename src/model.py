import tensorflow as tf

class ChurnModel:
    """
    Builds, compiles, and trains a neural network for churn prediction.
    """
    def __init__(self, input_shape):

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        
        return self.model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs, batch_size=batch_size, 
            callbacks=[early_stopping]
        )

    def evaluate(self, X_test, y_test):
        
        return self.model.evaluate(X_test, y_test)