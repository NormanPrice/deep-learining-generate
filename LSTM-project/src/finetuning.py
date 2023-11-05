
def fine_tune_on_words(model, words, char_indices, indices_char, seq_length, total_chars, learning_rate, epochs):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    fine_tune_text = ' '.join(words)
    X_fine, y_fine = vectorize_text(fine_tune_text, seq_length, total_chars, char_indices)
    history = model.fit(X_fine, y_fine, batch_size=32, epochs=epochs)
    return model