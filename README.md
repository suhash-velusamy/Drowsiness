# 🚗 Driver Drowsiness Detection System

A comprehensive Computer Vision and Deep Learning framework designed to monitor driver attentiveness in real-time and prevent road accidents caused by fatigue and drowsiness. 

This repository provides:
1. **🌐 Interactive Web Dashboard (`app.py`)**: A modern browser-based telemetry application with real-time video streaming, live Chart.js graphs, Web Audio synthesizer alarms, fatigue scoring, sensitivity sliders, and trip report CSV export.
2. **Real-Time Landmark-Based Detection (`live_detection.py`)**: High-performance, lightweight live monitoring using **MediaPipe Face Mesh** and dynamic **Eye Aspect Ratio (EAR)** calculation with immediate auditory alerts.
3. **Deep Learning Hybrid Pipeline (CNN + LSTM)**: A spatio-temporal deep learning model utilizing **InceptionV3 transfer learning** for feature extraction combined with an **LSTM recurrent neural network** for sequence classification.

---

## 📌 Features

- 🌐 **Modern Web Telemetry UI**: Dark-mode glassmorphism dashboard with real-time video streaming, blink rates, and fatigue metrics.
- 👁️ **Real-Time Eye Aspect Ratio (EAR) & MAR Tracking**: Calculates geometric eye openness and yawning ratio across 3D facial landmarks.
- 🔊 **Dual Audio Alert System**: In-browser Web Audio synthesizer alarm + backend buzzer when drowsiness threshold is crossed.
- 📈 **Live Chart.js Telemetry Graph**: 30-second rolling window showing instantaneous eye and mouth ratios.
- 🎛️ **Live Sensitivity Tuning**: Adjust EAR threshold, alarm delay, and yawn parameters on the fly via UI sliders.
- 📋 **Incident Logging & Trip Report**: Complete safety score calculation, incident logs, and 1-click CSV report export.
- 🧠 **Deep Feature Extraction**: InceptionV3 bottleneck feature extraction (2048-dim vectors) for sequence modeling.
- ⏳ **Temporal Sequence Learning**: Multi-frame LSTM network capturing temporal patterns of drowsiness over frame sequences.

---

## 🛠️ Tech Stack & Requirements

- **Programming Language**: Python 3.8+
- **Computer Vision**: OpenCV (`cv2`), MediaPipe
- **Deep Learning**: TensorFlow, Keras
- **Data & Math**: NumPy, Pandas, H5Py, Natsort
- **Operating System**: Windows (sound alerts via `winsound`), Linux/macOS compatible

---

## 📁 Repository Structure

```
DriverDrowsinessDetection/
├── live_detection.py        # Real-time webcam drowsiness detection (MediaPipe + EAR + Audio Alarm)
├── models.py                # Keras LSTM neural network architecture definition
├── train_test.py            # Train & evaluate the LSTM model on extracted feature sequences
├── data.py                  # Dataset management class for batch loading .npy sequences
├── extract_features.py      # Extract 2048-dim InceptionV3 features from face video frames
├── face_detection.py        # CamShift & Haar Cascade face detection and frame crop utility
├── csv_file_creator.py      # Automated dataset structuring and train/test CSV generator
├── retrain.py               # Transfer learning script to retrain InceptionV3 bottleneck graph
├── run_model.sh             # Shell script automating the end-to-end training pipeline
├── face_landmarker.task     # MediaPipe 3D Face Landmarker model binary
├── rm.model.json            # Model architecture export (JSON)
├── result_file.csv          # Evaluation and prediction log output
└── input/                   # Raw input video dataset directory
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/DriverDrowsinessDetection.git
cd DriverDrowsinessDetection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

Install dependencies:

```bash
pip install opencv-python mediapipe tensorflow keras numpy pandas matplotlib natsort h5py
```

---

## 💻 Usage

### 1. Launch the Interactive Web Dashboard 🌐

To start the full web application with real-time video streaming, live graphs, and incident analytics:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

- **Features in Web UI**:
  - Click **Start Camera Stream** to launch the video feed.
  - View real-time **Eye Aspect Ratio (EAR)**, **Mouth Ratio (MAR)**, and **Blink Rate**.
  - Adjust sensitivity sliders (**EAR threshold**, **Alarm delay**, **MAR threshold**) directly in the UI.
  - Hear synthesized audio alarms in your browser on drowsiness detection.
  - Review trip statistics & click **"Download Full Report (CSV)"**.

---

### 2. Standalone Real-Time Live Detection (Desktop OpenCV Window)

Run the standalone live detector via your webcam:

```bash
python live_detection.py
```

- **How it works**:
  1. Automatically downloads the `face_landmarker.task` MediaPipe model if not present.
  2. Tracks 3D facial landmarks for both left and right eyes.
  3. Computes average Eye Aspect Ratio (EAR).
  4. Triggers a red bounding box & audio alarm (`Beep`) when eyes remain closed (`EAR < 0.20`) for over 0.8 seconds.
- Press `q` to quit the application window.

---

### 2. Deep Learning Pipeline (CNN + LSTM Training)

To build, train, and test the spatio-temporal deep learning model:

#### Step 1: Face Detection & Frame Extraction
Extract cropped face ROI frames from input video dataset:
```bash
python face_detection.py 1 input/1/Alert_Video1.mp4 ./Output/1/Alert/ 1000
```

#### Step 2: Dataset Preparation
Split raw dataset into training/testing partitions and create directory structures:
```bash
python csv_file_creator.py 0.8
```

#### Step 3: Transfer Learning (InceptionV3)
Fine-tune InceptionV3 graph def on extracted facial images:
```bash
python retrain.py --output_graph=./output_graph.pb --output_labels=./output_labels.txt --image_dir=./data/training/
```

#### Step 4: Bottleneck Feature Extraction
Extract 2048-dimensional feature vectors into temporal `.npy` sequences:
```bash
python extract_features.py 50
```

#### Step 5: LSTM Training & Evaluation
Train the LSTM model on extracted feature sequences:
```bash
python train_test.py 50 1000
```

---

## 🔬 Model Architecture

### LSTM Classifier (`models.py`)
- **Input Shape**: `(sequence_length, 2048)`
- **LSTM Layer**: 2048 units with 50% Dropout
- **Dense Layers**: 1024 units (ReLU) ➔ 512 units (ReLU) ➔ Dropout (0.5)
- **Output Layer**: 2 classes (`Alert`, `Drowsy`) with Softmax activation
- **Optimizer**: Adam (`learning_rate=0.00005`), Loss: Categorical Crossentropy

---

## 📄 References & Acknowledgments

- Landmark detection powered by [Google MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
- Transfer learning based on InceptionV3 architecture.
- Inspired by research on temporal video sequence classification for driver fatigue monitoring.
