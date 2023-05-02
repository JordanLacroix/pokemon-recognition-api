# Pokémon Recognition API

This project is a Flask API for Pokémon recognition from images. It uses a deep learning model trained with TensorFlow/Keras to identify Pokémon in images sent via HTTP POST requests.

## Prerequisites

1. Install Python 3.9 or later:

- [Download Python](https://www.python.org/downloads/)

2. Install and configure Conda:

- [Download and install Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

3. Configure TensorFlow for your environment:

- [Install TensorFlow with Conda](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/)

Depending on your system, you might need to install additional packages or drivers to use TensorFlow with GPU support. Please refer to the [TensorFlow GPU support documentation](https://www.tensorflow.org/install/gpu) for more information.

## Installation

1. Clone the GitHub repository:

```
git clone https://github.com/user/pokemon_recognition.git
```

## Usage

1. Launch the Flask API locally inside Conda environment with Tensorflow:

```
python app.py
```

2. In a separate terminal, send a POST request to the API with a Pokémon image to recognize:

```
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:5000/predict
```

3. The API will return a JSON response containing the recognized Pokémon name:

```
{
  "pokemon": "pikachu"
}
```

## Project Structure

- `app.py`: The main file containing the Flask application and prediction functions.
- `pokemon_recognition_model.h5`: The file containing the pre-trained deep learning model for Pokémon recognition.
- `train/`: The directory containing the training images for the model.
