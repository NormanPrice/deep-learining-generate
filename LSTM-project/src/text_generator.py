

# src/text_generator.py
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sample(preds: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample an index from a probability array, with optional temperature.

    Parameters:
    - preds: np.ndarray - The prediction probabilities.
    - temperature: float - The temperature for sampling (higher temperature increases diversity).

    Returns:
    - int - The index of the sampled element.
    """
    try:
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    except ValueError as e:
        logger.exception("Invalid predictions array or temperature for sampling.")
        raise

def generate_text(model, length: int, diversity: float, text: str, seq_length: int, char_indices: dict, indices_char: dict) -> str:
    """
    Generate text of a specified length from a model, starting with a random seed.

    Parameters:
    - model: trained model capable of prediction.
    - length: int - The number of characters to generate.
    - diversity: float - The diversity for sampling.
    - text: str - The text to use for seeding.
    - seq_length: int - The length of sequences used by the model.
    - char_indices: dict - Mapping of characters to indices.
    - indices_char: dict - Mapping of indices to characters.

    Returns:
    - str - The generated text.
    """
    try:
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
    except Exception as e:
        logger.exception("An error occurred during text generation.")
        raise

