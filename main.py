import cv2
from camera import get_frame, release_camera
from detector import detect_faces

while True:
    ret, frame = get_frame()

    if not ret:
        break

    faces = detect_faces(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Live Face App", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

release_camera()
cv2.destroyAllWindows()