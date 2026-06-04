import cv2

def simple_swap(frame, selected_face_img, target_face):
    x1, y1, x2, y2 = target_face.bbox.astype(int)

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width <= 0 or face_height <= 0:
        return frame

    resized_face = cv2.resize(selected_face_img, (face_width, face_height))

    frame[y1:y2, x1:x2] = resized_face

    return frame