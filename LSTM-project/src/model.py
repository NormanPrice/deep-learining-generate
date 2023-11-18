# src/model.py
'''
from keras.models import Sequential, load_model, save_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

def create_model(seq_length, total_chars, load_saved_model, pretrained_model_path):
    if load_saved_model:
        return load_model(pretrained_model_path)
    else:
        model = Sequential()
        model.add(LSTM(256, input_shape=(seq_length, total_chars), return_sequences=True))
        model.add(LSTM(512, return_sequences=True))
        model.add(LSTM(512))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(total_chars, activation='softmax'))
        optimizer = Adam(learning_rate=0.00001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

def train_model(model, X, y):
    history = model.fit(X, y, batch_size=32, epochs=100)
    model.save('/models/model-saveLSTM.h5')
    plt.plot(history.history['loss'])
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
'''

# src/model.py
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import os
import logging

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
        return model
    except Exception as e:
        logger.exception("Failed to create or load the model", exc_info=e)
        raise

def train_model(model, X, y, model_save_path):
    """
    Trains the LSTM model with the given data.

    Parameters:
    - model: The LSTM model to be trained.
    - X: The input data for training.
    - y: The target data for training.
    - model_save_path: The path to save the trained model.

    Returns:
    - The history of the training process.
    """
    try:
        history = model.fit(X, y, batch_size=32, epochs=100)
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        return history
    except Exception as e:
        logger.exception("Failed to train the model", exc_info=e)
        raise


'''
MODEL WITH ATTENTION MECHANISMS - TO BE TRIED
model = Sequential()

# Bidirectional LSTMs with L1 regularization
model.add(Bidirectional(LSTM(256, input_shape=(seq_length, total_chars), kernel_regularizer=regularizers.l1(0.001))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(512, kernel_regularizer=regularizers.l1(0.001))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(512, kernel_regularizer=regularizers.l1(0.001))))

# Attention layer
model.add(AttentionLayer())

# Dense layers with dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(total_chars, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
'''
