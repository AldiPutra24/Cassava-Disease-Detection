from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the pre-trained model
classifier = hub.KerasLayer('https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2')

# Map the class names to human readable names
class_names = ['cmd', 'cbb', 'cgm', 'cbsd', 'healthy', 'unknown']
name_map = dict(
    cmd='Mosaic Disease',
    cbb='Bacterial Blight',
    cgm='Green Mite',
    cbsd='Brown Streak Disease',
    healthy='Healthy',
    unknown='Unknown')

remedies = dict(
    cmd='Use virus-free planting material and resistant varieties. Apply appropriate insecticides to control the whitefly population.',
    cbb='Apply copper-based bactericides and ensure good field sanitation. Remove and destroy affected plants.',
    cgm='Use resistant varieties and apply appropriate acaricides to control mite populations.',
    cbsd='Use disease-free planting materials and resistant varieties. Remove and destroy infected plants immediately.',
    healthy='No action needed. Keep up with regular plant care and monitoring.',
    unknown='Consult an expert for further diagnosis and management.'
)

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['file']
        if not file:
            return redirect(request.url)
        
        # Save the uploaded image
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Load and preprocess the image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img)

        # Predict the disease
        probabilities = classifier(img)
        prediction = tf.argmax(probabilities, axis=-1)
        predicted_class = class_names[prediction[0]]
        predicted_disease = name_map[predicted_class]
        remedy = remedies[predicted_class]

        # Pass the results to the template
        return render_template('result.html', image_url=file_path, disease=predicted_disease, remedy=remedy)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)