import cv2
import numpy as np


def extract_features(landmarks):
    """
    Extracts translation and scale-invariant 3D features from 21 MediaPipe hand landmarks.
    
    Returns a 1D numpy array of 63 features:
      (x_i - wrist_x) / scale, (y_i - wrist_y) / scale, (z_i - wrist_z) / scale
    """
    # Landmark 0 is the wrist
    wrist = landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    coords = []
    distances = []

    for lm in landmarks:
        dx = lm.x - wrist_x
        dy = lm.y - wrist_y
        dz = lm.z - wrist_z
        coords.extend([dx, dy, dz])
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(dist)

    # Scale normalization: divide by maximum distance from wrist
    max_dist = max(distances) if max(distances) > 1e-6 else 1.0
    normalized_features = np.array(coords, dtype=np.float32) / max_dist

    return normalized_features


def draw_styled_box(img, text, pos, bg_color=(30, 30, 30), text_color=(255, 255, 255), font_scale=0.7, thickness=2, padding=8):
    """
    Draws a readable text box with a solid background and border on the image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos

    x1 = max(0, x - padding)
    y1 = max(0, y - text_h - padding)
    x2 = min(img.shape[1] - 1, x + text_w + padding)
    y2 = min(img.shape[0] - 1, y + baseline + padding)

    # Background rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    # Border
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)
    # Text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
