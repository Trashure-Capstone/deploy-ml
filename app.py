from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
import pickle
from io import BytesIO
import keras.utils as image
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model('NewData_Fixed_MobileNetV2_1e-4_80Split_Adam_Dense_with02Dropout.h5')

class_names = class_names = ["hdpe_container", "hdpe_botol", "ps", "pp_tutup_botol", "pp_botol", "pp_sedotan", "pet", "pp_container", "ldpe_bag", "ldpe_botol", "hdpe_tutup_botol"]

# Load the model
# model_file_path = "Model.h5"
# with open(model_file_path, "rb") as file:
#     model = pickle.load(file)

# Preprocess input image
def preprocess_input(image):
    img_height, img_width = (200, 200)
    image = image.resize((img_height, img_width))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr.astype('float32') / 255
    return input_arr

# Predict image label
def predict_label(image):
    input_arr = preprocess_input(image)
    result = model.predict(input_arr)
    class_names = ['class_1', 'class_2', 'class_3', ...]  # Replace with your class names
    predicted_label = class_names[np.argmax(result)]
    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_contents = file.read()
            file_stream = BytesIO(file_contents)

            image = tf.keras.preprocessing.image.load_img(file_stream, target_size=(200,200))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])
            input_arr = input_arr.astype('float32') / 255
            result = model.predict(input_arr)
            print(result)
            print(class_names[np.argmax(result)])
        
        
        # if file:
        #     image = Image.open(file)
        #     predicted_label = predict_label(image)
        #     return jsonify({'predicted_label': predicted_label})
        # else:
        #     return "No file"
    # return render_template('index.html')
            return class_names[np.argmax(result)]
    if request.method == "GET":
        return "hasil"
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080))
