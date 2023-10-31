from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(seq_length, total_chars):
    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_length, total_chars), return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(total_chars, activation='softmax'))
    optimizer = Adam(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    #accuracy?
    #model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
