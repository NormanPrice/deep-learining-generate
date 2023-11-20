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

'''
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
        history = model.fit(X, y, batch_size=32, epochs=3)
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        return history
    except Exception as e:
        logger.exception("Failed to train the model", exc_info=e)
        raise

import numpy as np
from sklearn.model_selection import train_test_split

# Assuming 'X' and 'y' are already defined as per your provided code

# Define the split size for training and validation sets
train_size = 0.8  # 80% for training

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=42)

# Now, X_train and y_train are your training set
# X_val and y_val are your validation set

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity.
)

# Learning Rate Scheduling
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.97,          # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=3,          # Number of epochs with no improvement after which learning rate will be reduced.
    verbose=1,
    min_lr=0.0001        # Lower bound on the learning rate.
)

# Model training
history = model.fit(
    X, y,
    epochs=15,  # Total number of epochs
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, lr_scheduler]  # Add callbacks to training
)

model_save_path = '/kaggle/working/model-save-universal.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

plt.plot(history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
loss_plot_path = '/kaggle/working/model_loss_plot.png'
plt.savefig(loss_plot_path)
print(f"Loss plot saved to {loss_plot_path}")
plt.show()
'''
