def remove_non_ascii_printable(file_path):
    """
    Remove non-ASCII printable characters from a text file.

    :param file_path: Path to the text file to be processed.
    :return: True if the file was processed successfully, False otherwise.
    """
    try:
        with open(file_path, 'r') as file:
            text = file.read()

        # Filter out non-ASCII printable characters (ASCII range: 32 to 126 inclusive)
        filtered_text = ''.join(char for char in text if 32 <= ord(char) <= 126)

        with open(file_path, 'w') as file:
            file.write(filtered_text)

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Example usage:
file_path = '/Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/data/10-million-password-list-top-100000.txt'
result = remove_non_ascii_printable(file_path)
if result:
    print("Non-ASCII printable characters have been removed.")
else:
    print("An error occurred during processing.")

