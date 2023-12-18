

# src/model.py
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import os
import logging
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(seq_length, total_chars, load_saved_model=False, pretrained_model_path=None):
    """
    Creates or loads a LSTM model for character prediction.

    Parameters:
    - seq_length: The length of the sequence to be considered for prediction.
    - total_chars: The total number of unique characters in the text.
    - load_saved_model: Boolean flag indicating whether to load a saved model.
    - pretrained_model_path: The path to the saved model.

    Returns:
    - A Keras Sequential model.
    """
    try:
        if load_saved_model and pretrained_model_path:
            model = load_model(pretrained_model_path)
            logger.info(f"Model loaded from {pretrained_model_path}")
        else:
            model = Sequential()
            model.add(LSTM(256, input_shape=(seq_length, total_chars), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(512, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(512))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(total_chars, activation='softmax'))

            optimizer = Adam(learning_rate=0.0001)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            logger.info("New model created")
            print("MODEL SUMMARY:")
            print(model.summary())
        return model
    except Exception as e:
        logger.exception("Failed to create or load the model", exc_info=e)
        raise


def train_model(model, X, y, X_val, y_val, epochs, batch_size, model_save_path):
    """
    Trains the LSTM model with the given data.

    Parameters:
    - model: The LSTM model to be trained.
    - X: The input data for training.
    - y: The target data for training.
    - X_val, y_val: The validation data.
    - model_save_path: The path to save the trained model.

    Returns:
    - The history of the training process.
    """
    try:
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )

        # Learning Rate Scheduling
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.97,
            patience=3,
            verbose=1,
            min_lr=0.0001
        )

        # Model training with callbacks
        history = model.fit(
            X, y,
            epochs,  # You can adjust the number of epochs
            batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, lr_scheduler]
        )

        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        return history
    except Exception as e:
        logger.exception("Failed to train the model", exc_info=e)
        raise
