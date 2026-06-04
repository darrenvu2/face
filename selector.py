import cv2
import os

FACES_DIR = "faces"

def load_faces():
    faces = {}

    for file in os.listdir(FACES_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(FACES_DIR, file)
            faces[file] = cv2.imread(path)

    return faces