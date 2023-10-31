from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def train_and_save_model(model, X, y, model_path, total_epochs, batch_size):
    history = model.fit(X, y, batch_size=batch_size, epochs=total_epochs)
    model.save(model_path)
    plt.plot(history.history['loss'])
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
