import cv2
import os
from tqdm import tqdm

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------- FRAME EXTRACTION ----------

def extract_frames(video_path, save_dir, interval=3, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            cv2.imwrite(f"{save_dir}/frame_{saved}.jpg", frame)
            saved += 1

        count += 1

    cap.release()


# ---------- FACE DETECTION (HAAR) ----------

def crop_face(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return

    # take largest face
    x, y, w, h = sorted(
        faces, key=lambda x: x[2] * x[3], reverse=True
    )[0]

    face_crop = img[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (224, 224))

    cv2.imwrite(save_path, face_crop)


# ---------- MAIN PIPELINE ----------

def process_folder(input_videos, label):

    frame_dir = f"frames/{label}"
    face_dir = f"processed/{label}"

    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)

    idx = 0

    for video in tqdm(os.listdir(input_videos)):
        video_path = os.path.join(input_videos, video)

        extract_frames(video_path, frame_dir)

        for img in os.listdir(frame_dir):
            crop_face(
                os.path.join(frame_dir, img),
                f"{face_dir}/face_{idx}.jpg"
            )
            idx += 1

        # cleanup frames
        for img in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, img))


print("🔄 Preprocessing started...")

process_folder("dataset_raw/real", "real")
process_folder("dataset_raw/fake", "fake")

print("✅ Preprocessing finished")
