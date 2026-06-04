import cv2
from camera import get_frame, release_camera
from insight import detect_faces
from selector import load_faces
from swapper import simple_swap

source_faces = load_faces()
face_names = list(source_faces.keys())
selected_index = 0

while True:
    ret, frame = get_frame()

    if not ret:
        break

    detected_faces = detect_faces(frame)

    for face in detected_faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if face_names:
        selected_name = face_names[selected_index]
        cv2.putText(
            frame,
            f"Selected: {selected_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Live Face App", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("n") and face_names:
        selected_index = (selected_index + 1) % len(face_names)

release_camera()
cv2.destroyAllWindows()