# Real-Time Fatigue Monitoring System

A real-time drowsiness and fatigue detection system using live webcam input.  

This project includes two main modules:
- 👁️ **Eye Detection** – Detects if the driver’s eyes are closed
- 😮 **Yawn Detection** – Detects if the driver is yawning

It supports both:
- ✅ Custom CNN models
- ✅ Pretrained ResNet18 models

---

## 🚀 Live Demo (Streamlit App)

To run the **real-time demo**, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/AswinBalajiTR/Real-Time-Fatigue-Monitoring-System
```

```bash
# 2. Install all dependencies
pip install -r requirements.txt
```

```bash
# 3. Launch the app
streamlit run Main.py
```

---

## Project Retraining

To replicate the project or retrain the model, follow these steps:

### 1) Download the Dataset

Download the dataset zip file from the following link:

https://drive.google.com/file/d/1PSWj2w2LP6Zza125W4ZmCL7t8ozEnPlA/view

---

### 2) Unzip the Dataset

After downloading, unzip the data file using this command:

```bash
unzip data.zip
```

---

### 3) Run the Train Python Files

We have **different files for Eye Detection and Yawn Detection.**

####  3a. Eye Detection

```bash
# CNN Model
python3 train_eye_detection_CNN.py
```

```bash
# ResNet18 Model
python3 train_eye_detection_ResNet18.py
```

#### 3b. Yawn Detection

```bash
# CNN Model
python3 train_yawn_detection_ResNet18.py
```

```bash
# ResNet18 Model
python3 train_yawn_detection_CNN.py
```

---

## 🖼️ View the Model Metrics

✅ After running any of the above files, you will be able to see **the performance metrics** of each model in your terminal (e.g., Accuracy, Precision, Recall, F1-Score).

---

