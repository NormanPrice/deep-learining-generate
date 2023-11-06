# src/main.py
'''
from preprocessing import load_text, create_char_mappings, vectorize_text
from model import create_model, train_model
from text_generator import generate_text
from utils import save_text
import config

if __name__ == "__main__":
    text = load_text(config.filename)
    char_indices, indices_char, chars = create_char_mappings(text)
    total_chars = len(chars)
    
    X, y = vectorize_text(text, config.seq_length, total_chars, char_indices)
    model = create_model(config.seq_length, total_chars, load_saved_model=True)
    
    if config.train_model:
        train_model(model, X, y)
    
    generated_text = generate_text(model, 300, 0.5, text, config.seq_length, char_indices, indices_char)
    save_text(generated_text, config.output_path)
    print(type(X))
    print(generated_text)
'''

# src/main.py
import logging
from preprocessing import load_text, create_char_mappings, vectorize_text
from model import create_model, train_model
from text_generator import generate_text
from utils import save_text
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to execute the text generation process.
    """
    try:
        # Load the text from the specified file in the config
        text = load_text(config.filename)
        # Create character mappings from the text
        char_indices, indices_char, chars = create_char_mappings(text)
        total_chars = len(chars)
        
        # Vectorize the text as per the configurations
        X, y = vectorize_text(text, config.seq_length, total_chars, char_indices)
        # Create the model with the given configurations
        model = create_model(config.seq_length, total_chars, load_saved_model=config.load_saved_model)
        
        # If training is enabled in the configuration, train the model
        if config.train_model:
            train_model(model, X, y)
        
        # Generate text using the model
        generated_text = generate_text(model, config.gen_text_length, config.gen_text_temperature, text, config.seq_length, char_indices, indices_char)
        # Save the generated text to the specified output path
        save_text(generated_text, config.output_path)

        logger.info(f"Generated text:\n{generated_text}")
    
    except Exception as e:
        logger.exception("An unexpected error occurred during the text generation process:", exc_info=e)

if __name__ == "__main__":
    main()
