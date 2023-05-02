from flask import Flask, request, jsonify
import os
import io
import numpy as np
import json
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

def load_pokemon_model(model_path):
    """
    Load the Pokemon recognition model from the given file path.

    :param model_path: The file path to the model.
    :return: The loaded model.
    """
    model = load_model(model_path)
    return model

def predict_pokemon(img, model, class_indices):
    """
    Predict the class of the Pokemon in the given image using the model.

    :param img: The image to predict.
    :param model: The loaded model.
    :param class_indices: The dictionary mapping class names to their indices.
    :return: The predicted class of the Pokemon.
    """
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_label = list(class_indices.keys())[predicted_class]
    return class_label

def load_image_from_request(request):
    """
    Load the image from the given HTTP request and convert it to a pixel array.

    :param request: The HTTP request object.
    :return: The pixel array representing the image.
    """
    if 'file' not in request.files:
        return None
    file = request.files['file']
    if file.filename == '':
        return None
    img_bytes = file.read()
    img = image.load_img(io.BytesIO(img_bytes), target_size=(150, 150))
    img_array = image.img_to_array(img)
    return img_array

def load_class_indices(file_path):
    """
    Load the class indices from a JSON file.

    :param file_path: The file path to the JSON file containing the class indices.
    :return: A dictionary mapping class names to their indices.
    """
    with open(file_path, 'r') as f:
        class_indices = json.load(f)
    return class_indices

@app.route('/predict', methods=['POST'])
def predict():
    """
    This function is called when a POST HTTP request is received on the /predict path. 
    It loads the Pokémon recognition model, the image from the request, and uses the model 
    to predict the class of the Pokémon in the image. The response is returned in JSON 
    format containing the predicted class.
    """

    # Load the model, class indices, and image from the request
    model_path = 'pokemon_recognition_model.h5'
    class_indices = load_class_indices('class_indices.json')
    model = load_pokemon_model(model_path)
    img = load_image_from_request(request)

    # If no image was loaded, return an error
    if img is None:
        return jsonify({'error': 'No file in request'})

    # Predict the class of the Pokémon in the image
    predicted_pokemon = predict_pokemon(img, model, class_indices)
    return jsonify({'pokemon': predicted_pokemon})

if __name__ == '__main__':
    app.run(debug=True)
