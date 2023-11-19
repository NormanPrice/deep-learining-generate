def remove_non_ascii_printable(file_path):
    """
    Remove non-ASCII printable characters from a text file while preserving its structure.

    :param file_path: Path to the text file to be processed.
    :return: True if the file was processed successfully, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Filter out non-ASCII printable characters, but keep line breaks and other whitespaces
        # ASCII range for printable characters: 32 to 126
        # ASCII codes for newline (10) and carriage return (13)
        filtered_text = ''.join(char for char in text if 32 <= ord(char) <= 126 or ord(char) in [10, 13])

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(filtered_text)

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

'''# Example usage:
file_path = '/Users/norbert/Downloads/deep-learining-generate-main/LSTM-project/data/darkweb2017.txt'
result = remove_non_ascii_printable(file_path)
if result:
    print("Non-ASCII printable characters have been removed.")
else:
    print("An error occurred during processing.")
'''
