import os
import time
import warnings
import cv2
import mediapipe as mp
import pandas as pd
from utils import extract_features, draw_styled_box

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def ask_labels():
    print("=" * 60)
    print("      ASL HAND SIGN DATA COLLECTOR      ")
    print("=" * 60)
    print("Enter the letters or signs you want to record, separated by commas.")
    print("Press ENTER directly to record ALL letters (A to Z)!")
    print("-" * 60)
    user_input = input("Enter signs to collect [default: ALL (A-Z)]: ").strip().upper()

    if not user_input or user_input == "ALL":
        labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    else:
        labels = [s.strip() for s in user_input.split(",") if s.strip()]

    samples_input = input("Number of samples per sign [default: 100]: ").strip()
    num_samples = int(samples_input) if samples_input.isdigit() and int(samples_input) > 0 else 100

    return labels, num_samples


def main():
    labels, num_samples_per_class = ask_labels()
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0). Check camera connection/permissions.")
        return

    # Check if existing CSV exists
    collected_data = []
    feature_columns = [f"f_{i}" for i in range(63)]

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        for label in labels:
            print(f"\n[INFO] Next sign: '{label}'")
            print("  Press 's' in the webcam window to start recording.")
            print("  Press 'skip' by pressing 'n' to go to next sign.")
            print("  Press 'q' anytime to exit and save progress.")

            # Ready state: wait for user to press 's'
            ready = False
            while not ready:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Process hand landmarks for preview
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                # UI Prompts
                draw_styled_box(
                    frame,
                    f"TARGET SIGN: [ {label} ]",
                    (20, 50),
                    bg_color=(50, 20, 20),
                    text_color=(0, 255, 255),
                    font_scale=0.9,
                    thickness=2,
                )
                draw_styled_box(
                    frame,
                    "Show hand gesture, then press 'S' to start recording",
                    (20, 100),
                    bg_color=(20, 20, 20),
                    text_color=(255, 255, 255),
                    font_scale=0.6,
                )
                draw_styled_box(
                    frame,
                    "Press 'N' to skip | Press 'Q' to finish & save",
                    (20, 140),
                    bg_color=(20, 20, 20),
                    text_color=(180, 180, 180),
                    font_scale=0.5,
                )

                cv2.imshow("Sign Language Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') or key == ord('S'):
                    ready = True
                elif key == ord('n') or key == ord('N'):
                    print(f"[INFO] Skipped '{label}'")
                    break
                elif key == ord('q') or key == ord('Q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    save_data(collected_data, feature_columns)
                    return

            if not ready:
                continue

            # Countdown 3-2-1
            for count in [3, 2, 1]:
                start_time = time.time()
                while time.time() - start_time < 0.8:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    draw_styled_box(
                        frame,
                        f"RECORDING '{label}' IN {count}...",
                        (w // 2 - 160, h // 2),
                        bg_color=(0, 0, 150),
                        text_color=(0, 255, 255),
                        font_scale=1.0,
                    )
                    cv2.imshow("Sign Language Data Collector", frame)
                    cv2.waitKey(1)

            # Recording loop
            recorded_count = 0
            while recorded_count < num_samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                        features = extract_features(hand_landmarks.landmark)
                        collected_data.append([label] + features.tolist())
                        recorded_count += 1

                progress_text = f"Recording '{label}': {recorded_count}/{num_samples_per_class} frames"
                draw_styled_box(
                    frame,
                    progress_text,
                    (20, 50),
                    bg_color=(0, 100, 0),
                    text_color=(255, 255, 255),
                    font_scale=0.8,
                )
                draw_styled_box(
                    frame,
                    "Keep hand in frame and move slightly for variety",
                    (20, 90),
                    bg_color=(20, 20, 20),
                    text_color=(200, 200, 200),
                    font_scale=0.55,
                )

                cv2.imshow("Sign Language Data Collector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quitting early...")
                    break

    cap.release()
    cv2.destroyAllWindows()
    save_data(collected_data, feature_columns)


def save_data(new_records, columns):
    if not new_records:
        print("[INFO] No new records captured.")
        return

    new_df = pd.DataFrame(new_records, columns=["label"] + columns)

    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(CSV_PATH, index=False)
    print(f"\n[SUCCESS] Successfully saved dataset to '{CSV_PATH}'!")
    print(f"Total samples in dataset: {len(combined_df)}")
    print("Class counts:")
    print(combined_df["label"].value_counts())


if __name__ == "__main__":
    main()
