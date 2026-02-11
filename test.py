import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model

IMG_SIZE = 224

# load model
model = load_model("deepfake_mobilenet.h5")

# get image path from terminal
if len(sys.argv) < 2:
    print("Usage: python3 test.py <image_path>")
    exit()

img_path = sys.argv[1]

img = cv2.imread(img_path)

if img is None:
    print("❌ Could not read image:", img_path)
    exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

print("Probability:", pred)

if pred > 0.5:
    print("Fake")
else:
    print("Real")

