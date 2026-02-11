# Deepfake Detection System

A deep learning based system to detect fake images and videos.

## Features
- Face extraction using MTCNN
- CNN (MobileNetV2) classifier
- Frame preprocessing
- Image testing
- Automatic face detection testing

## Setup
pip install -r requirements.txt

## Run

Preprocess:
python3 preprocess.py

Train:
python3 train_model.py

Test image:
python3 auto_test.py image.jpg
