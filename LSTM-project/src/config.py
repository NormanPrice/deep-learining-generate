import sys, os
seq_length = 20
filename = "/Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/data/rockyou-75.txt"
model_save_path = "models/model-saveLSTM-91-chars.h5"
output_path = "outputs/LSTM-output-new.txt"
load_saved_model=True
train_model = False
chars_path = "/Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/data/ascii_printable_characters.txt"
epochs = 10
batch_size = 64
