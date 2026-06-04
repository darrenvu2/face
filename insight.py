from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(320, 320))

def detect_faces(frame):
    return app.get(frame)