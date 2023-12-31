# LSTM Password Generator

## Description
This project implements a console application that allows users to train and use LSTM (Long Short-Term Memory) deep learning models for generating passwords. It includes functionalities such as text comparison, ASCII checking, and model implementation for password generation.

## Files and Functionality
- **compare.py**: Contains functions for reading and comparing sets of words from files. Used to determine the overlap between passwords generated by the model and actual passwords datasets (used for training/validation).
- **main-args.py**: Main script handling command-line arguments.
- **config.py**: Configuration settings for the project.
- **model.py**: Contains model definitions and training parameters.
- **preprocessing.py**: Script for data preprocessing.
- **text_generator.py**: Involved in generating new passwords.
- **utils.py**: Utility functions supporting various tasks across the project.

## Usage
This app allows users to specify parameters for training via command-line arguments.

### Sample Usage for Training New Model:
python3 main-args.py --train_model --filename /path/to/dataset/ --output_path /path/ --save_model_path /path/ --gen_text_length /length/ --gen_text_temperature /temperature/ --batch_size /batch_size/ --epochs /epochs/

### Sample Usage for Generating Passwords Using Pre-trained Model:
python3 main-args.py --output_path /path/ --pretrained_model_path /path/to/model/


## Configuration
The configuration file includes the following values:
- `seq_length = 20`
- `filename = "/LSTM-project/data/rockyou-75.txt"`
- `model_save_path = "models/model-saveLSTM.h5"`
- `output_path = "outputs/LSTM-output-new.txt"`
- `load_saved_model = True`
- `train_model = False`
- `chars_path = "/LSTM-project/data/ascii_printable_characters.txt"`
- `epochs = 10`
- `batch_size = 64`

### Model Details
    The LSTM model to be trained with default parameters is outlined as follows:
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm (LSTM)                 (None, 20, 256)           361472    
                                                                     
     dropout (Dropout)           (None, 20, 256)           0         
                                                                     
     lstm_1 (LSTM)               (None, 20, 512)           1574912   
                                                                     
     dropout_1 (Dropout)         (None, 20, 512)           0         
                                                                     
     lstm_2 (LSTM)               (None, 512)               2099200   
                                                                     
     dense (Dense)               (None, 512)               262656    
                                                                     
     dropout_2 (Dropout)         (None, 512)               0         
                                                                     
     dense_1 (Dense)             (None, 96)                49248     
                                                                     
    =================================================================
    Total params: 4347488 (16.58 MB)
    Trainable params: 4347488 (16.58 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________

    An LSTM layer with 256 units, taking input of shape (seq_length, total_chars), and returning sequences.
    A Dropout layer with a dropout rate of 0.2.
    Another LSTM layer with 512 units, returning sequences.
    A second Dropout layer with a dropout rate of 0.2.
    A third LSTM layer with 512 units.
    A Dense layer with 512 units and ReLU activation.
    A third Dropout layer with a dropout rate of 0.3.
    A final Dense layer with total_chars units and softmax activation.

The model is compiled with categorical crossentropy loss and an Adam optimizer with a learning rate of 0.0001.

**Note:**
- The `(None, 20, 256)` output shape represents the `seq_length` value (20).
- The `(None, 96)` output shape of the last layer indicates the number of characters used to train the model (96).

Due to above - in case you want to generate passwords using pre-tranied model, please ensure that you have both model file and dataset on which the model was trained - so that dimensions of the model are correct(num_chars might vary - that will cause errors).

Sample error caused by mismatch in char numbers:
    `ValueError: Input 0 of layer "sequential_2" is incompatible with the layer: expected shape=(None, 25, 83), found shape=(None, 25, 81)` -  In this case Model was tranied on 83 chars and the input for processing here is 81 -> causing mismatch

This will be worked on in the future by making sure that all models are tranied on ASCII printable charachter set.
Sorry for the temporary inconvinience :(


### Training parameters and funcionality

**Training Parameters:**

    model (Keras model): The LSTM model to be trained.
    X (array): The input data for training.
    y (array): The target data for training.
    X_val, y_val (arrays): The validation data.
    epochs (int): Number of epochs for training.
    batch_size (int): Batch size for training.
    model_save_path (str): The path to save the trained model.

**Training Functionality:**

Train function implements early stopping, monitoring validation loss with a patience of 5 epochs and restoring the best weights upon stopping.
It also uses a learning rate scheduler to reduce the learning rate by a factor of 0.97 with a patience of 3 epochs, based on validation loss, with a minimum learning rate set to 0.0001.
The model is trained with the specified X, y, X_val, y_val, epochs, and batch_size, using the defined callbacks for early stopping and learning rate scheduling.

