import cv2
import sys
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("deepfake_mobilenet.h5")
detector = MTCNN()

img_path = sys.argv[1]
img = cv2.imread(img_path)

faces = detector.detect_faces(img)

if len(faces) == 0:
    print("No face detected")
    exit()

x,y,w,h = faces[0]['box']
face = img[y:y+h, x:x+w]

face = cv2.resize(face,(IMG_SIZE,IMG_SIZE))
face = face/255.0
face = np.expand_dims(face,0)

pred = model.predict(face)[0][0]

print("Probability:", pred)

if pred>0.5:
    print("Fake")
else:
    print("Real")
