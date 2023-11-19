# deep-learining-generate
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

**Note:**
- The `(None, 20, 256)` output shape represents the `seq_length` value (20).
- The `(None, 96)` output shape of the last layer indicates the number of characters used to train the model (96).
