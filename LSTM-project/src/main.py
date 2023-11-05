# src/main.py
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
