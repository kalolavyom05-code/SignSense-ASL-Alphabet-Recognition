# SignSense 🤟
### Real-Time American Sign Language (ASL) Alphabet Recognition Using Machine Learning

SignSense is a Python-based machine learning project that recognizes American Sign Language (ASL) alphabet hand gestures through a webcam in real time.

It uses **MediaPipe** to detect hand landmarks, **OpenCV** for webcam processing, and a **Random Forest classifier** to predict alphabet gestures.

The application also includes prediction smoothing and an interactive sentence-building interface.

## ✨ Features

- **Real-Time Hand Tracking:** Detects 21 hand landmarks using MediaPipe.
- **Gesture Recognition:** Recognizes trained ASL alphabet hand gestures.
- **Machine Learning:** Uses a Scikit-learn Random Forest classifier.
- **Custom Dataset Collection:** Collect gesture samples using your webcam.
- **Prediction Smoothing:** Uses majority voting across recent predictions.
- **Confidence Filtering:** Filters predictions below the configured confidence threshold.
- **Interactive Sentence Builder:** Build sentences using recognized letters.
- **CPU-Based Processing:** Runs using a webcam without requiring a dedicated GPU.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam and image processing |
| MediaPipe | Hand landmark detection |
| Scikit-learn | Random Forest classification |
| NumPy | Numerical operations |
| Pandas | Dataset management |

## 📁 Project Structure

```text
SignSense-ASL-Alphabet-Recognition/
│
├── data/
│   ├── landmarks.csv
│   └── landmarks_backup.csv
│
├── model/
│   └── model.p
│
├── collect_data.py
├── create_sample_data.py
├── train_classifier.py
├── realtime_inference.py
├── utils.py
│
├── collect.bat
├── train.bat
├── run.bat
│
├── requirements.txt
├── .gitignore
├── README.md
└── PRESENTATION_GUIDE.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SignSense-ASL-Alphabet-Recognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Follow these three steps to use SignSense.

### Step 1: Collect Hand Gesture Data

```bash
python collect_data.py
```

- Select the alphabet letters you want to record.
- Choose the number of samples per letter.
- Show your hand gesture to the webcam.
- Press `S` to start recording.
- Press `N` to skip a letter.
- Press `Q` to finish and save progress.

### Step 2: Train the Machine Learning Model

```bash
python train_classifier.py
```

The training program:

- Loads the collected landmark dataset.
- Splits the data into 80% training and 20% testing.
- Trains a Random Forest classifier with 100 estimators.
- Evaluates predictions using accuracy and classification reports.
- Saves the trained model to `model/model.p`.

### Step 3: Start Real-Time Recognition

```bash
python realtime_inference.py
```

- Allow webcam access.
- Show your hand gesture to the camera.
- Hold a recognized gesture steadily to add its letter.
- Use the keyboard controls to build sentences.

### Keyboard Controls

| Key | Function |
|---|---|
| Hold gesture | Add predicted letter |
| Space | Insert space |
| Backspace | Delete last letter |
| C | Clear sentence |
| Q | Quit application |

## 🧠 Machine Learning Workflow

1. Capture hand gestures using a webcam.
2. Detect hand landmarks using MediaPipe.
3. Extract normalized hand landmark features.
4. Store gesture features in a CSV dataset.
5. Train a Random Forest classifier.
6. Evaluate the trained model.
7. Load the saved model for real-time predictions.
8. Apply temporal smoothing to stabilize predictions.

## 📊 Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Estimators | 100 |
| Training split | 80% |
| Testing split | 20% |
| Random state | 42 |
| Features | 63 |
| Prediction history | 5 frames |
| Confidence threshold | 0.40 |

Actual recognition performance depends on the dataset, lighting, camera quality, and hand positioning. Results should be evaluated on unseen data.

## 🔮 Future Improvements

- Expand gesture vocabulary.
- Improve recognition under different lighting conditions.
- Add support for dynamic gestures.
- Develop a graphical user interface.
- Add text-to-speech output.
- Explore multilingual interfaces.
- Improve model generalization using larger datasets.

## ⚠️ Project Scope

SignSense currently focuses on recognizing trained ASL alphabet hand gestures. It is not a complete American Sign Language translator and does not interpret ASL grammar, facial expressions, or full conversations.

## 👨‍💻 Project Information

**Project:** SignSense

**Category:** Machine Learning / Computer Vision

**Language:** Python

**Application:** Real-Time ASL Alphabet Recognition

**Author:** [Your Name]

## 📜 Credits and Licensing

Original-source attribution and licence compatibility must be verified before publishing. Any adapted third-party code should be credited according to its original licence.

