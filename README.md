## Project Documentation

### Overview
This project involves training a Deep Convolutional Generative Adversarial Network (DCGAN) to generate EEG signals. The project includes a Tkinter-based GUI application for entering DCGAN parameters.

### Project Structure
- `gui.py`: Contains the Tkinter-based GUI application.
- `train.py`: Contains the training logic for the GAN.
- `generator.py`: Defines the `Generator1D` class.
- `discriminator.py`: Defines the `Discriminator1D` class.
- `initialization.py`: Contains functions for initializing weights.
- `dataset.py`: Contains the `EEGDataset` class.
- `utils.py`: Contains utility functions like `load_eeg_data`, `plot_signal`, and `plot_signal_to_image`.

### Requirements
- Python 3.x
- PyTorch
- TensorBoard
- PIL (Pillow)
- Tkinter

### Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

#### Tkinter GUI Application
1. Run the Tkinter GUI application:
   ```sh
   python gui.py
   ```

2. Use the GUI to upload a MATLAB file and start the training process.

#### GUI Parameters
- **Learning Rate**: The rate at which the model learns during training. A lower value means slower learning.
- **Batch Size**: The number of samples processed before the model is updated.
- **Noise Dimension**: The size of the random noise vector input to the generator.
- **Number of Epochs**: The number of complete passes through the training dataset.
- **Features (Discriminator)**: The number of features in the discriminator network.
- **Features (Generator)**: The number of features in the generator network.


### Training
The training process involves loading EEG data, initializing the generator and discriminator, and training them using the specified parameters. The training progress is logged using TensorBoard.

