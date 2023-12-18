
# src/preprocessing.py
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_text(filename: str) -> str:
    """
    Load text from a given filename.

    Parameters:
    - filename: str - Path to the text file to be loaded.

    Returns:
    - str - The content of the file.
    """
    try:
        with open(filename, encoding='utf-8-sig') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.exception(f"The file {filename} was not found.")
        raise
    except Exception as e:
        logger.exception(f"An error occurred while loading the file {filename}.")
        raise

def create_char_mappings(text: str) -> (dict, dict, list):
    """
    Create mappings from characters to indices and vice versa.

    Parameters:
    - text: str - The text for which character mappings are created.

    Returns:
    - Tuple containing two dictionaries (char to index, index to char) and a list of unique characters.
    """
    chars = sorted(set(text))
    char_indices = {c: i for i, c in enumerate(chars)}
    indices_char = {i: c for i, c in enumerate(chars)}
    return char_indices, indices_char, chars

def vectorize_text(text: str, seq_length: int, total_chars: int, char_indices: dict) -> (np.ndarray, np.ndarray):
    """
    Vectorize text into input (X) and output (y) matrices for model training.

    Parameters:
    - text: str - The text to vectorize.
    - seq_length: int - The length of the sequences to be created.
    - total_chars: int - The number of unique characters in the text.
    - char_indices: dict - A dictionary mapping characters to indices.

    Returns:
    - Tuple of Numpy arrays: input matrix (X) and output matrix (y).
    """
    try:
        # Using np.bool_ to save memory as each entry is only True or False
        X = np.zeros((len(text) - seq_length, seq_length, total_chars), dtype=np.bool_)
        y = np.zeros((len(text) - seq_length, total_chars), dtype=np.bool_)

        for i in range(len(text) - seq_length):
            for t, char in enumerate(text[i:i + seq_length]):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[text[i + seq_length]]] = 1

        return X, y
    except Exception as e:
        logger.exception("An error occurred during vectorization of the text.")
        raise
