def count_common_words(smaller_file_path, larger_file_path):
    # Function to read file and return a set of words
    def read_file_to_set(file_path):
        with open(file_path, 'r') as file:
            return set(file.read().lower().split())

    # Read words from both files
    smaller_file_words = read_file_to_set(smaller_file_path)
    larger_file_words = read_file_to_set(larger_file_path)

    # Find common words
    common_words = smaller_file_words.intersection(larger_file_words)

    # Return the count of common words
    return len(common_words)

# Example usage
smaller_file_path = '/Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/outputs/newoutput_1000_25.txt'
larger_file_path = '/Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/data/10-million-password-list-top-100000.txt'
common_word_count = count_common_words(smaller_file_path, larger_file_path)
print(f'Number of common words: {common_word_count}')
