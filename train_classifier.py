import os
import pickle
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("data", "landmarks.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.p")


def train():
    print("=" * 60)
    print("        TRAINING SIGN LANGUAGE CLASSIFIER       ")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found at '{DATA_PATH}'.")
        print("Please run 'python collect_data.py' first to collect gesture samples.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded dataset with {len(df)} samples across {df['label'].nunique()} classes.")

    if df["label"].nunique() < 2:
        print("[ERROR] You need at least 2 different classes to train a classifier.")
        print("Classes found:", list(df["label"].unique()))
        print("Please collect data for at least one more sign.")
        return

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Train / Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] Test set size: {len(X_test)} samples")
    print("[INFO] Training Random Forest model...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"             MODEL ACCURACY: {acc * 100:.2f}%")
    print("=" * 60)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "classes": list(model.classes_)}, f)

    print(f"[SUCCESS] Model successfully saved to '{MODEL_PATH}'!")
    print("You can now run 'python realtime_inference.py' to test it live with your webcam.")


if __name__ == "__main__":
    train()
