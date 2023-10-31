def save_generated_text(text, output_file):
    with open(output_file, 'w') as file:
        file.write(text)
