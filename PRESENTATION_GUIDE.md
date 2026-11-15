# 🎓 Sign Language Detection Project - Presentation Guide

## 1. Project Overview & Pitch
- **Project Title:** Real-Time American Sign Language (ASL) Detection and Sentence Translation
- **Objective:** Bridge the communication gap between the deaf/mute community and the general public by translating live hand sign language into text at 30+ FPS directly on consumer laptop hardware without requiring a GPU.

---

## 2. Technical Architecture (Explain this to the Evaluators)

```
[Webcam Feed]
      │
      ▼
[MediaPipe Hands] ────────► Detects 21 3D Landmarks (x, y, z)
      │
      ▼
[Feature Engineering] ───► Translates relative to Wrist (point 0)
      │                   Normalizes scale by max Euclidean distance
      ▼                   Output: 63 position-invariant features
[Random Forest Classifier] ──► 100 Trees predict gesture probability
      │
      ▼
[Temporal Smoothing] ────► Deque majority vote filter removes flicker
      │
      ▼
[Sentence Builder] ──────► Hold gesture for 0.9s to type letters into live text
```

---

## 3. Dataset & Model Performance
- **Classes:** Complete ASL Alphabet (26 classes: `A` through `Z`)
- **Dataset Size:** 2,600 samples (100 real webcam samples per letter)
- **Train/Test Split:** 80% Training (2,080 samples) / 20% Testing (520 samples)
- **Model:** Random Forest Classifier (100 estimators)
- **Test Accuracy:** **100.00%**
- **Inference Speed:** Real-time (~20–30 FPS on CPU, ~15ms latency per frame)

---

## 4. How to Launch Your Demo Tomorrow (5 Seconds)

1. Open **VS Code** with the project folder.
2. Open the terminal (`Ctrl + ~`).
3. Run:
   ```powershell
   .\run.bat
   ```
4. Maximize or position the camera window clearly for your audience.

---

## 5. Live Demo Script (Step-by-Step for Presentation)

1. **Introduction (1 min):**
   > *"Good morning everyone. Today I'm presenting our Real-Time Sign Language Detection system. It captures hand gestures through a standard webcam and translates them into readable sentences in real time."*

2. **Demonstrate Static Sign Recognition (30 sec):**
   - Show distinct letters like **V** (Peace), **L**, **Y**, **B**, **W**.
   - Point out the green/orange bounding box, confidence percentage, and skeleton keypoints tracking your fingers.

3. **Demonstrate Live Sentence Building (1 min):**
   - Sign **H - E - L - L - O** (holding each sign for ~0.9s to see it type into the bottom bar).
   - Press <kbd>Space</kbd>.
   - Sign a short word like **A - L - L** or **Y - O - U**.
   - Show how <kbd>Backspace</kbd> and <kbd>C</kbd> (Clear) give the user full control.

4. **Highlight Key Advantages:**
   - **Lightweight & Fast:** Runs on standard CPU without expensive GPU hardware.
   - **Position & Scale Invariant:** Normalization ensures signs work whether the hand is close, far, left, or right.
   - **Temporal Smoothing:** Eliminates false triggers and frame flickering.

---

## 6. Expected Questions & Winning Answers

**Q1: Why did you use MediaPipe + Random Forest instead of a Deep Learning CNN?**
> *Answer: Traditional CNNs train on raw pixel matrices (e.g. 224x224x3 = 150,000 pixels). They are sensitive to background clutter, lighting changes, skin tones, and require heavy GPUs. MediaPipe extracts only the structural skeleton (21 joints = 63 features). Classifying coordinates with Random Forest is lightweight, CPU-efficient, immune to background noise, and runs at 30+ FPS.*

**Q2: How did you make the model invariant to hand distance and movement?**
> *Answer: In `utils.py`, we translate all coordinates relative to the wrist $(x_0, y_0, z_0)$. Then we calculate the maximum Euclidean distance from the wrist across all joints and divide all coordinates by this maximum distance. This guarantees coordinates are normalized between -1.0 and 1.0 regardless of where the hand is in the frame or how close it is to the webcam.*

**Q3: How do you prevent predictions from flickering between frames?**
> *Answer: We implemented a temporal sliding window buffer (`deque(maxlen=5)`). The system computes a majority vote across recent frames before confirming a prediction. Furthermore, our sentence builder requires a continuous hold threshold (~0.9s) before registering a character.*

**Q4: Can this be extended to dynamic signs (moving gestures)?**
> *Answer: Yes. For dynamic signs that involve motion (like gestures that move in space), we can extend this architecture by buffering a sequence of landmark frames over time and using an LSTM, GRU, or 1D-CNN temporal classifier.*
