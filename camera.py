import cv2

cap = cv2.VideoCapture(0)

def get_frame():
    return cap.read()

def release_camera():
    cap.release()