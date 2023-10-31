import numpy as np

def load_and_clean_text(filename, seq_length):
    with open(filename, encoding='utf-8-sig') as f:
        text = f.read().strip()

    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    X = np.zeros((len(text) - seq_length, seq_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(text) - seq_length, len(chars)), dtype=np.bool)
    for i in range(len(text) - seq_length):
        for t, char in enumerate(text[i:i + seq_length]):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[text[i + seq_length]]] = 1

    return X, y, chars, char_indices, indices_char
