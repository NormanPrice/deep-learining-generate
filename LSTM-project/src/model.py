# src/model.py
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
