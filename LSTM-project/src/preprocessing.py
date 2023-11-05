# src/preprocessing.py
import numpy as np

def load_text(filename):
    with open(filename, encoding='utf-8-sig') as f:
        return f.read().strip()

def create_char_mappings(text):
    chars = sorted(set(text))
    char_indices = {c: i for i, c in enumerate(chars)}
    indices_char = {i: c for i, c in enumerate(chars)}
    return char_indices, indices_char, chars

def vectorize_text(text, seq_length, total_chars, char_indices):
    X = np.zeros((len(text) - seq_length, seq_length, total_chars), dtype=np.bool)
    y = np.zeros((len(text) - seq_length, total_chars), dtype=np.bool)
    for i in range(len(text) - seq_length):
        for t, char in enumerate(text[i:i + seq_length]):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[text[i + seq_length]]] = 1
    return X, y
