from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os

app = Flask(__name__)

# Lazy load model (Render → tidak timeout)
classifier = None

def load_model():
    global classifier
    if classifier is None:
        classifier = hub.KerasLayer(
            'https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2'
        )
    return classifier

class_names = ['cmd', 'cbb', 'cgm', 'cbsd', 'healthy', 'unknown']
name_map = dict(
    cmd='Mosaic Disease',
    cbb='Bacterial Blight',
    cgm='Green Mite',
    cbsd='Brown Streak Disease',
    healthy='Healthy',
    unknown='Unknown'
)

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
        file = request.files.get('file')
        if not file:
            return redirect(request.url)

        # Ensure static folder exists
        os.makedirs('static', exist_ok=True)
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img)

        model = load_model()
        probabilities = model(img)
        prediction = tf.argmax(probabilities, axis=-1)

        pred_id = prediction.numpy()[0]
        pred_class = class_names[pred_id]
        predicted_disease = name_map[pred_class]
        remedy = remedies[pred_class]

        return render_template(
            'result.html',
            image_url=file_path,
            disease=predicted_disease,
            remedy=remedy
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
