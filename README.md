# Cassava Disease Detection 🌿
This repository contains a deep learning project aimed at detecting and classifying diseases in cassava plants using image data. Cassava, a crucial crop in many regions, is vulnerable to several diseases that can significantly affect yield. Early and accurate detection of these diseases is essential for effective management and control.

## Overview

This project utilizes a pre-trained model to classify cassava leaves into one of several categories, including:
- **Mosaic Disease (CMD)**
- **Bacterial Blight (CBB)**
- **Green Mite (CGM)**
- **Brown Streak Disease (CBSD)**
- **Healthy**

The model is hosted on TensorFlow Hub, and the classification process involves uploading an image of a cassava leaf, predicting the disease, and providing recommended remedies for management.

## Features

- **Accurate Classification**: The model predicts cassava leaf diseases with high accuracy.
- **Pre-trained Model**: Utilizes a TensorFlow Hub model, ensuring reliable performance with minimal setup.
- **Actionable Remedies**: Provides recommendations for managing the detected disease.
- **User-Friendly Interface**: Simple script to upload images and get instant predictions.

## Installation

   ```bash
   git clone https://github.com/AldiPutra24/cassava-disease-detection.git

   cd Cassava-Disease-Detection

   py -3 -m venv .venv

   .venv\Scripts\activate

   pip install flask tensorflow tensorflow-hub tensorflow-datasets matplotlib opencv-python

   python app.py
   ```

<br />

## References
This project is based on the [TensorFlow Hub CropNet Cassava Disease Classifier tutorial](https://www.tensorflow.org/hub/tutorials/cropnet_cassava), which provides a comprehensive guide to using the CropNet model for cassava disease detection.
