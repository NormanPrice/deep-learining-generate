# src/utils.py
def save_text(text, filepath):
    with open(filepath, 'w') as file:
        file.write(text)
