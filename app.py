from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf
import json


app = Flask(__name__)



# Load model
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights('model_weights.h5')

# Define class names
class_names = ['class_1', 'class_2', 'class_3']  # Define your class names here

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/script.js')
def send_js():
    return send_file('script.js')


@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from request
    image_data = request.json['image_data']

    # Decode base64 image data and preprocess
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    image = image.resize((150, 150))  # Resize the image to match the model's input shape
    image = image.convert("RGB")  # Convert image to RGB mode
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict class of the image
    predictions = loaded_model.predict(image)
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    return jsonify({'prediction': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
