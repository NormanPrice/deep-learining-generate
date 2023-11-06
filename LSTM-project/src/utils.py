# src/utils.py
'''
def save_text(text, filepath):
    with open(filepath, 'w') as file:
        file.write(text)
'''

# src/utils.py
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_text(text: str, filepath: str) -> None:
    """
    Save a string of text to a specified file path.

    Parameters:
    - text: str - The text to be saved.
    - filepath: str - The path to the file where the text should be saved.

    Raises:
    - IOError: If the file could not be written to the specified path.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(text)
        logger.info(f"Text successfully saved to {filepath}")
    except IOError as e:
        logger.exception(f"Failed to save text to {filepath}")
        raise

