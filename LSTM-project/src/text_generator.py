# src/text_generator.py
import numpy as np

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, length, diversity, text, seq_length, char_indices, indices_char):
    start_index = np.random.randint(0, len(text) - seq_length - 1)
    generated = ''
    sentence = text[start_index: start_index + seq_length]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, seq_length, len(indices_char)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        generated += next_char
    return generated
