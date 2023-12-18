import argparse
from preprocessing import load_text, create_char_mappings, vectorize_text
from model import create_model, train_model 
from text_generator import generate_text
from utils import save_text
from sklearn.model_selection import train_test_split
import config
from check_ascii import remove_non_ascii_printable

def main(args):
    # Load and preprocess text


    text_ascii = load_text(config.chars_path)        
    char_indices, indices_char, chars = create_char_mappings(text_ascii)
    total_chars = len(chars)
    
    remove_non_ascii_printable(args.filename)
    text = load_text(args.filename)
    #char_indices, indices_char, chars = create_char_mappings(text)
    #total_chars = len(chars)
    
    # Vectorize text
    X, y = vectorize_text(text, args.seq_length, total_chars, char_indices)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # Create or load model
    model = create_model(args.seq_length, total_chars, load_saved_model=bool(args.pretrained_model_path), pretrained_model_path=args.pretrained_model_path)
    
    # Print model summary if a pre-trained model is loaded
    if args.pretrained_model_path:
        print("Loaded model summary:")
        model.summary()

    # Train the model
    if args.train_model:
        model = train_model(model, X_train, y_train, X_val, y_val, args.batch_size, args.epochs, args.save_model_path)

        # Save the model if specified
        if args.save_model_path:
            model.save(args.save_model_path)
    
    # Generate text
    generated_text = generate_text(model, args.gen_text_length, args.gen_text_temperature, text, args.seq_length, char_indices, indices_char)
    save_text(generated_text, args.output_path)
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation script")
    parser.add_argument("--filename", type=str, default='/Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/data/ascii_printable_characters.txt', help="Path to the text file to be processed - containing all ASCII prontable charachters")
    #parser.add_argument("--char_filename", type=str, required=True, help="Path to the text file to be processed")
    parser.add_argument("--seq_length", type=int, default=config.seq_length, help="Sequence length for training")
    parser.add_argument("--train_model", action="store_true", help="Flag to train the model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated text")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to a pre-trained model")
    parser.add_argument("--save_model_path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--gen_text_length", type=int, default=300, help="Length of text to generate")
    parser.add_argument("--gen_text_temperature", type=float, default=0.5, help="Temperature for text generation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")

    args = parser.parse_args()
    
    main(args)

#python3 main-args.py --train_model --filename /Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/data/darkweb2017.txt --output_path /Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/outputs/output.txt --save_model_path /Users/norbert/Downloads/deep-learining-generate-main/models/model-test.h5
#hit_ratio = sampled_passwords_in_test_set / all_sampled_passwords
