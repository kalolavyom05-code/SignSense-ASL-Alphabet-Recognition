import os
import time
import pickle
import warnings
from collections import deque, Counter
import cv2
import mediapipe as mp
import numpy as np
from utils import extract_features, draw_styled_box

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join("model", "model.p")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def main():
    print("=" * 60)
    print("      REAL-TIME SIGN LANGUAGE DETECTOR      ")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model file '{MODEL_PATH}' was not found.")
        print("Please follow these steps first:")
        print("  1. Run 'python collect_data.py' to record your signs")
        print("  2. Run 'python train_classifier.py' to train the model")
        return

    # Load trained model
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        classes = data["classes"]

    print(f"[INFO] Loaded model with {len(classes)} classes: {classes}")
    print("[CONTROLS]")
    print("  - Hold gesture steady for 1.2s: Adds letter to sentence")
    print("  - SPACE bar: Add a space")
    print("  - BACKSPACE: Delete last letter")
    print("  - C key: Clear sentence")
    print("  - Q key: Quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # Prediction smoothing and sentence building variables
    recent_predictions = deque(maxlen=5)
    current_sentence = ""
    current_held_char = None
    char_hold_start_time = 0
    HOLD_THRESHOLD_SEC = 0.9
    CONFIDENCE_THRESHOLD = 0.40

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            predicted_label = None
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Extract coordinates and bounding box
                    x_list = [lm.x for lm in hand_landmarks.landmark]
                    y_list = [lm.y for lm in hand_landmarks.landmark]
                    xmin = max(0, int(min(x_list) * w) - 20)
                    xmax = min(w, int(max(x_list) * w) + 20)
                    ymin = max(0, int(min(y_list) * h) - 20)
                    ymax = min(h, int(max(y_list) * h) + 20)

                    # Feature extraction and prediction
                    features = extract_features(hand_landmarks.landmark).reshape(1, -1)
                    probabilities = model.predict_proba(features)[0]
                    max_idx = np.argmax(probabilities)
                    best_conf = probabilities[max_idx]
                    best_label = model.classes_[max_idx]

                    if best_conf >= CONFIDENCE_THRESHOLD:
                        recent_predictions.append(best_label)
                    else:
                        recent_predictions.append(None)

                    # Majority vote across recent frames
                    valid_preds = [p for p in recent_predictions if p is not None]
                    if valid_preds:
                        most_common_label, count = Counter(valid_preds).most_common(1)[0]
                        if count >= 2:
                            predicted_label = most_common_label
                            confidence = best_conf

                    # Draw bounding box and label
                    if predicted_label:
                        box_color = (0, 220, 100) if confidence >= 0.60 else (0, 180, 255)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                        label_text = f"{predicted_label} ({confidence * 100:.0f}%)"
                        draw_styled_box(
                            frame,
                            label_text,
                            (xmin, max(30, ymin - 10)),
                            bg_color=(20, 20, 20),
                            text_color=(0, 255, 120) if confidence >= 0.60 else (0, 200, 255),
                            font_scale=0.8,
                            thickness=2,
                        )
                    else:
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (150, 150, 150), 2)

            # Hold-to-type logic for sentence building
            now = time.time()
            if predicted_label:
                if predicted_label == current_held_char:
                    held_duration = now - char_hold_start_time
                    if held_duration >= HOLD_THRESHOLD_SEC:
                        current_sentence += predicted_label
                        current_held_char = None  # Reset so it doesn't repeatedly append
                        char_hold_start_time = now
                else:
                    current_held_char = predicted_label
                    char_hold_start_time = now
            else:
                current_held_char = None

            # Calculate FPS
            fps_frame_count += 1
            if now - fps_start_time >= 1.0:
                fps = fps_frame_count / (now - fps_start_time)
                fps_frame_count = 0
                fps_start_time = now

            # Top Bar: FPS & Instructions
            draw_styled_box(
                frame,
                f"FPS: {fps:.1f}",
                (w - 120, 30),
                bg_color=(30, 30, 30),
                text_color=(200, 200, 200),
                font_scale=0.55,
                thickness=1,
            )

            # Typing progress indicator
            if current_held_char and (now - char_hold_start_time) > 0.3:
                progress = min(1.0, (now - char_hold_start_time) / HOLD_THRESHOLD_SEC)
                bar_w = int(150 * progress)
                cv2.rectangle(frame, (20, 75), (20 + bar_w, 85), (0, 255, 255), -1)
                draw_styled_box(
                    frame,
                    f"Holding: {current_held_char}",
                    (20, 65),
                    bg_color=(30, 30, 30),
                    text_color=(0, 255, 255),
                    font_scale=0.55,
                    thickness=1,
                )

            # Bottom Sentence Bar
            cv2.rectangle(frame, (0, h - 60), (w, h), (20, 20, 20), -1)
            cv2.line(frame, (0, h - 60), (w, h - 60), (80, 80, 80), 1)
            if current_sentence:
                sentence_display = f"Sentence: {current_sentence}"
                text_col = (0, 255, 120)
            else:
                sentence_display = "Sentence: (Hold a sign steady to type)"
                text_col = (140, 140, 140)

            cv2.putText(
                frame,
                sentence_display,
                (20, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                text_col,
                2,
                cv2.LINE_AA,
            )

            # Controls help
            cv2.putText(
                frame,
                "[Space]: Space | [Bksp]: Del | [C]: Clear | [Q]: Quit",
                (w - 480, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Sign Language Recognition", frame)

            # Keyboard handler
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == 32:  # SPACE
                current_sentence += " "
            elif key == 8:   # BACKSPACE
                current_sentence = current_sentence[:-1]
            elif key == ord('c') or key == ord('C'):
                current_sentence = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
