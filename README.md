# 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system built using **Convolutional Neural Networks (CNN)** and **Computer Vision**. The model detects whether a driver's eyes are open or closed from a webcam feed and triggers an alarm when drowsiness is detected.

---

## 📌 Demo

📁 **Dataset:** [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)

---

## 🧠 How It Works

1. **Face Detection** — Haar Cascade detects the driver's face in each frame
2. **Eye Detection** — Eye regions are extracted from the face ROI
3. **CNN Prediction** — Each eye ROI is passed to the trained CNN model
4. **Drowsiness Logic** — If eyes remain closed for 15+ consecutive frames, an alarm triggers
5. **Alarm** — Audio alert plays and stops only after the driver opens their eyes

---

## 📁 Project Structure

```
Driver-Drowsiness-Detection-System/
│
├── dataset/
│   ├── train/
│   │   ├── open/
│   │   └── closed/
│   └── test/
│       ├── open/
│       └── closed/
│
├── Drowsiness_Detection.ipynb   # Full pipeline notebook
├── drowsiness_model.h5          # Trained CNN model
├── alarm.wav                    # Alert sound
└── README.md
```

---

## 📊 Model Performance

### Training History

![Training History](/Output/training_history.png)

The model converges quickly — validation accuracy reaches **~98%** within the first few epochs. Training and validation curves are closely aligned, indicating no significant overfitting.

### Confusion Matrix

![Confusion Matrix](/Output/confusion_matrix.png)

|                   | Predicted Closed | Predicted Open |
| ----------------- | ---------------- | -------------- |
| **Actual Closed** | 8246 ✅          | 158 ❌         |
| **Actual Open**   | 156 ❌           | 8435 ✅        |

### Classification Report

| Class        | Precision | Recall | F1-Score | Support    |
| ------------ | --------- | ------ | -------- | ---------- |
| Closed       | 0.98      | 0.98   | 0.98     | 8,404      |
| Open         | 0.98      | 0.98   | 0.98     | 8,591      |
| **Accuracy** |           |        | **0.98** | **16,995** |
| Macro Avg    | 0.98      | 0.98   | 0.98     | 16,995     |
| Weighted Avg | 0.98      | 0.98   | 0.98     | 16,995     |

> ✅ **Test Accuracy: 98.15%** &nbsp;|&nbsp; 📉 **Test Loss: 0.0489**

---

## 🏗️ CNN Architecture

```
Input (24×24 grayscale)
    │
    ├── Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ├── Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ├── Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    │
    ├── Flatten
    ├── Dense(256) → BatchNorm → Dropout(0.5)
    └── Dense(1, sigmoid)  →  open / closed
```

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam (lr=1e-3)
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/DebasishBisoi524/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. run the full notebook

Open `Drowsiness_Detection.ipynb` in Jupyter and run all cells.

---

## 📦 Requirements

```
streamlit
tensorflow
opencv-python-headless
numpy
Pillow
pygame
```

---

## 📂 Dataset

The [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) contains **84,898 eye images** collected from 37 subjects under various lighting, reflective, and sensor conditions.

| Split | Open  | Closed | Total |
| ----- | ----- | ------ | ----- |
| Train | ~42K  | ~42K   | ~84K  |
| Test  | ~8.6K | ~8.4K  | ~17K  |

---

## ⚙️ Real-Time Detection Features

- ✅ Face + eye detection using Haar Cascades
- ✅ Per-eye CNN prediction with confidence score
- ✅ Drowsiness counter with visual progress bar (green → yellow → red)
- ✅ Audio alarm with 7-second delay stop after eyes open
- ✅ Flashing red border alert when drowsy
- ✅ Exits cleanly via Q, ESC, or window X button
- ✅ FPS counter overlay

---

## 🙋 Author

**Debasish Bisoi**  
B.Tech Student | KIIT University  
🔗 [GitHub](https://github.com/DebasishBisoi524)

---

## 📄 License

This project is for educational purposes. The MRL Eye Dataset is credited to the [Machine Learning Research Lab, VSB-TUO](http://mrl.cs.vsb.cz).
