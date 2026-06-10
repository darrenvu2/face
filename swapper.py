import cv2
import numpy as np
from insight import detect_faces

def simple_swap(frame, selected_face_img, target_face):
    #detect the face in the source JPG
    source_faces = detect_faces(selected_face_img)

    if not source_faces:
        return frame
    
    #crop just the face region from the source image
    src_face = source_faces[0]
    sx1, sy1, sx2, sy2 = src_face.bbox.astype(int)

    # clamp to image bounds
    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(selected_face_img.shape[1], sx2)
    sy2 = min(selected_face_img.shape[0], sy2)

    cropped_face = selected_face_img[sy1:sy2, sx1:sx2]

    if cropped_face.size == 0:
        return frame

    # get the target region in the live frame
    tx1, ty1, tx2, ty2 = target_face.bbox.astype(int)

    tx1 = max(0, tx1)
    ty1 = max(0, ty1)
    tx2 = min(frame.shape[1], tx2)
    ty2 = min(frame.shape[0], ty2)

    face_width = tx2 - tx1
    face_height = ty2 - ty1

    if face_width <= 0 or face_height <= 0:
        return frame

    #resize cropped source face to match target size and paste
    resized_face = cv2.resize(cropped_face, (face_width, face_height))
    frame[ty1:ty2, tx1:tx2] = resized_face

    return frame