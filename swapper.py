# swapper.py
import insightface
from insight import detect_faces

swapper = insightface.model_zoo.get_model("inswapper_128.onnx")

def real_swap(frame, selected_face_img, target_face):
    source_faces = detect_faces(selected_face_img)

    if not source_faces:
        return frame

    source_face = source_faces[0]

    result = swapper.get(
        frame,
        target_face,
        source_face,
        paste_back=True
    )

    return result