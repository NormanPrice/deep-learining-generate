import argparse
import data_preparation
import model
import generation
import train
import generate
from tensorflow.keras.models import load_model

def main(args):
    X, y, chars, char_indices, indices_char = data_preparation.load_and_clean_text(args.filename, args.seq_length)
    total_chars = len(chars)

    if args.load_saved_model:
        lstm_model = load_model(args.model_path)
        print("Model loaded from", args.model_path)
    else:
        lstm_model = model.create_model(args.seq_length, total_chars)
        print("New model created")

    if args.train_model:
        train.train_and_save_model(lstm_model, X, y, args.model_path, args.total_epochs, args.batch_size)
        print("Model trained and saved at", args.model_path)

    generated_text = generation.generate_text(lstm_model, X, args.seq_length, char_indices, indices_char, args.generate_length, args.diversity)
    print("Generated Text:")
    print(generated_text)
    generate.save_generated_text(generated_text, args.output_file)
    print("Generated text saved at", args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Model and Generate Passwords")
    parser.add_argument("--filename", type=str, required=True, help="Path to the text file for training")
    parser.add_argument("--model_path", type=str, default="model-saveLSTM.h5", help="Path to save/load the model")
    parser.add_argument("--output_file", type=str, default="LSTM-output.txt", help="Path to save the generated text")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length for training")
    parser.add_argument("--total_epochs", type=int, default=100, help="Total epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--generate_length", type=int, default=1000, help="Length of the generated text")
    parser.add_argument("--diversity", type=float, default=0.5, help="Diversity for text generation")
    parser.add_argument("--load_saved_model", action="store_true", help="Flag to load saved model")
    parser.add_argument("--train_model", action="store_true", help="Flag to train model")

    args = parser.parse_args()
    main(args)


#python main.py --filename /path/to/textfile.txt --train_model
#python main.py --filename /path/to/textfile.txt --model_path /path/to/saved/model.h5 --load_saved_model
#python main.py --filename /path/to/textfile.txt --model_path /path/to/save/model.h5 --total_epochs 50 --batch_size 16 --seq_length 30 --train_model
