import cv2
import numpy as np
from insight import detect_faces

def get_landmarks(face, img_shape):
    """Extract 5 key landmarks from insightface"""
    kps = face.kps.astype(np.float32)  # shape (5, 2): left eye, right eye, nose, left mouth, right mouth
    return kps

def warp_source_to_target(src_img, src_face, tgt_face, tgt_shape):
    """Warp source face to align with target face using landmark affine transform"""
    src_kps = src_face.kps.astype(np.float32)
    tgt_kps = tgt_face.kps.astype(np.float32)

    # Estimate affine transform from source landmarks -> target landmarks
    transform, _ = cv2.estimateAffinePartial2D(src_kps, tgt_kps)

    if transform is None:
        return None

    # Warp entire source image into target frame space
    warped = cv2.warpAffine(
        src_img,
        transform,
        (tgt_shape[1], tgt_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return warped

def get_face_mask(face, frame_shape):
    """Build a convex hull mask around the face landmarks + bbox"""
    kps = face.kps.astype(np.int32)
    x1, y1, x2, y2 = face.bbox.astype(int)

    # Add bbox corners around the face area to make the hull larger
    extra_pts = np.array([
        [x1, y1], [x2, y1], [x1, y2], [x2, y2],
        [(x1+x2)//2, y1], [(x1+x2)//2, y2],
        [x1, (y1+y2)//2], [x2, (y1+y2)//2],
    ], dtype=np.int32)

    all_pts = np.concatenate([kps, extra_pts], axis=0)
    hull = cv2.convexHull(all_pts)

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Soften mask edges
    mask = cv2.GaussianBlur(mask, (21, 21), 11)

    return mask, hull

def simple_swap(frame, selected_face_img, target_face):
    # 1. Detect face in source JPG
    source_faces = detect_faces(selected_face_img)
    if not source_faces:
        return frame

    src_face = source_faces[0]

    # 2. Warp source face to align landmarks with target face
    warped = warp_source_to_target(selected_face_img, src_face, target_face, frame.shape)
    if warped is None:
        return frame

    # 3. Build convex hull mask around the target face
    mask, hull = get_face_mask(target_face, frame.shape)

    # 4. Find center of face for seamlessClone
    x1, y1, x2, y2 = target_face.bbox.astype(int)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Clamp center to frame bounds
    h, w = frame.shape[:2]
    center = (
        max(1, min(w - 1, center[0])),
        max(1, min(h - 1, center[1]))
    )

    # 5. Seamless clone — blends lighting and color naturally
    try:
        result = cv2.seamlessClone(warped, frame, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # Fallback: simple alpha blend if seamlessClone fails
        mask_3ch = mask[:, :, np.newaxis] / 255.0
        result = (warped * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

    return result