# 🤟 ISL Recognition System

An end-to-end **Indian Sign Language (ISL) Recognition System** built using **MediaPipe**, **TensorFlow**, and **Deep Learning**.

The project extracts body and hand landmarks from sign language videos using MediaPipe, converts them into temporal landmark sequences, and trains an LSTM-based neural network for gesture recognition.

---

# Features

- Video-based ISL recognition
- MediaPipe Pose + Hand landmark extraction
- Handedness-aware landmark processing
- Landmark sequence generation
- LSTM-based deep learning model
- Training and evaluation pipeline
- Confusion matrix generation
- Classification report generation
- Dataset validation
- Label encoding
- Modular project architecture

---

# Project Structure

```text
ISL-Recognition-System
│
├── dataset/
│   ├── downloads/
│   ├── raw/
│   └── processed/
│
├── docs/
├── logs/
├── models/
├── notebooks/
├── outputs/
├── saved_models/
│
├── src/
│   ├── augmentation/
│   ├── evaluation/
│   ├── preprocessing/
│   ├── training/
│   ├── utils/
│   └── ...
│
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Model Pipeline

```text
Videos
    │
    ▼
MediaPipe Detection
    │
    ▼
Pose Landmarks (33)
Hand Landmarks (21 × 2)
    │
    ▼
258-D Feature Vector
    │
    ▼
Sequence Generator
(60 Frames)
    │
    ▼
LSTM Network
    │
    ▼
Predicted ISL Word
```

---

# Landmark Vector

Each frame contains:

| Component | Points | Values |
|----------|-------:|-------:|
| Pose | 33 | 132 |
| Left Hand | 21 | 63 |
| Right Hand | 21 | 63 |
| **Total** | | **258 Features** |

---

# Dataset

Current implementation uses the **INCLUDE Dataset**.

Current experiment:

- Classes : 6
- Videos : 85
- Sequence Length : 60
- Feature Size : 258

Classes:

- Summer
- Spring
- Winter
- Fall
- Season
- Monsoon

---

# Training

```bash
python -m src.training.train
```

---

# Evaluation

The evaluation module automatically generates:

- Test Accuracy
- Classification Report
- Confusion Matrix
- Evaluation JSON

---

# Technologies Used

- Python
- TensorFlow
- MediaPipe
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib

---

# Current Baseline

| Model | Accuracy |
|--------|---------:|
| LSTM | **52.94%** |

Dataset:

- 6 Classes
- 85 Videos

---

# Future Improvements

- Bidirectional LSTM
- Attention Mechanism
- Hyperparameter Optimization
- Temporal Data Augmentation
- Real-time Webcam Recognition
- GUI Application
- Full INCLUDE Dataset Training

---

# Version History

## v1.0.0

- Landmark extraction pipeline
- Dataset validation
- Sequence generation
- LSTM baseline
- Training pipeline
- Evaluation pipeline
- Confusion matrix
- Classification report

---

# Author

Soumya JS Nair

GitHub:
https://github.com/soumyapramod2012

---

# Acknowledgements

- MediaPipe
- TensorFlow
- INCLUDE Dataset
- OpenCV