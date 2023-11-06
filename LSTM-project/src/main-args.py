'''
# src/main.py
import argparse
from preprocessing import load_text, create_char_mappings, vectorize_text
from model import create_model, train_model, save_model
from text_generator import generate_text
from utils import save_text

def main(filename, seq_length, train_model_flag, output_path, pretrained_model_path, save_model_path):
    text = load_text(filename)
    char_indices, indices_char, chars = create_char_mappings(text)
    total_chars = len(chars)
    
    X, y = vectorize_text(text, seq_length, total_chars, char_indices)
    model = create_model(seq_length, total_chars, load_saved_model=bool(pretrained_model_path), pretrained_model_path=pretrained_model_path)
    
    if train_model_flag:
        model = train_model(model, X, y)
        if save_model_path:
            save_model(model, save_model_path)
    
    #generated_text = generate_text(model, 300, 0.5, text, seq_length, char_indices, indices_char)
    generated_text = generate_text(model, args.gen_text_length, args.gen_text_temperature, text, seq_length, char_indices, indices_char)
    save_text(generated_text, output_path)
    print(type(X))
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation script")
    parser.add_argument("--filename", type=str, required=True, help="Path to the text file to be processed")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length for training (default: 40)")
    parser.add_argument("--train_model", action="store_true", help="Flag to train the model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated text")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to a pre-trained model")
    parser.add_argument("--save_model_path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--gen_text_length", type=int, default=300, help="Length of text to generate (default: 300)")
    parser.add_argument("--gen_text_temperature", type=float, default=0.5, help="Temperature for text generation (default: 0.5)")
    #args = parser.parse_args()
    args = parser.parse_args()
    
    main(args.filename, args.seq_length, args.train_model, args.output_path, args.pretrained_model_path, args.save_model_path)

#norbert@Norberts-MacBook-Air src % python3 main-args.py --filename /Users/norbert/Desktop/LSTM/data/rockyou-75.txt --seq_length 20 --output_path /Users/norbert/Desktop/LSTM/outputs/LSTM-output-new.txt --pretrained_model_path /Users/norbert/Desktop/LSTM/models/model-saveLSTM-91-chars.h5
'''

# src/main.py
import argparse
import logging
import sys

from preprocessing import load_text, create_char_mappings, vectorize_text
from model import create_model, train_model, save_model
from text_generator import generate_text
from utils import save_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """
    The main function of the text generation script.
    """
    try:
        text = load_text(args.filename)
        char_indices, indices_char, chars = create_char_mappings(text)
        total_chars = len(chars)

        X, y = vectorize_text(text, args.seq_length, total_chars, char_indices)
        model = create_model(args.seq_length, total_chars, 
                             load_saved_model=bool(args.pretrained_model_path), 
                             pretrained_model_path=args.pretrained_model_path)

        if args.train_model:
            model = train_model(model, X, y)
            if args.save_model_path:
                save_model(model, args.save_model_path)

        generated_text = generate_text(
            model, args.gen_text_length, args.gen_text_temperature,
            text, args.seq_length, char_indices, indices_char
        )

        save_text(generated_text, args.output_path)
        logger.info(f"Generated text saved to {args.output_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation script.")
    parser.add_argument("--filename", type=str, required=True, help="Path to the text file to be processed")
    parser.add_argument("--seq_length", type=int, default=40, help="Sequence length for training (default: 40)")
    parser.add_argument("--train_model", action="store_true", help="Flag to train the model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated text")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to a pre-trained model")
    parser.add_argument("--save_model_path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--gen_text_length", type=int, default=300, help="Length of text to generate (default: 300)")
    parser.add_argument("--gen_text_temperature", type=float, default=0.5, help="Temperature for text generation (default: 0.5)")

    args = parser.parse_args()
    
    main(args)
