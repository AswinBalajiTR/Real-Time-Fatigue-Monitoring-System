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

## ⬇️ Download the Dataset

Download the dataset zip file from the following link:

https://drive.google.com/file/d/1PSWj2w2LP6Zza125W4ZmCL7t8ozEnPlA/view

---

## 📦 Unzip the Dataset

After downloading, unzip the data file using this command:

```bash
unzip data.zip
```

---

## 🚀 Run the Python Files

We have **different files for Eye Detection and Yawn Detection.**

---

### 👁️ Eye Detection

#### ✅ Run the Custom CNN Model

```bash
python3 eye_detection.py
```

#### ✅ Run the Pretrained ResNet18 Model

```bash
python3 train_eye_detection_ResNet18.py
```

---

### 😮 Yawn Detection

#### ✅ Run the Main Yawn Detection Model

```bash
python3 train_yawn_detection_ResNet18.py
```

#### ✅ Run the Baseline Yawn Detection Model

```bash
python3 train_yawn_detection_CNN.py
```

---

## 🖼️ View the Model Metrics

✅ After running any of the above files, you will be able to see **the performance metrics** of each model in your terminal (e.g., Accuracy, Precision, Recall, F1-Score).

---

