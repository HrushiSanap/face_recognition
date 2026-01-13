# Smart Face Recognition System with Liveness Detection

A robust, modular, and privacy-focused face recognition system built with Python, OpenCV, and Dlib. This project goes beyond simple matching by implementing **Smart Liveness Detection** (Blink + Head Movement tracking) to prevent photo spoofing attacks.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)  
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)  
![License](https://img.shields.io/badge/License-Unlicense-lightgrey?style=for-the-badge)

## 📖 Table of Contents
- [About the Project](#-about-the-project)
- [Directory Structure](#-directory-structure)
- [Key Features](#-key-features)
- [Technical Logic (Liveness)](#-technical-logic-how-it-works)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🔭 About the Project

This system is designed to identify users in real-time while ensuring they are physically present. Unlike standard face recognition scripts that can be tricked by holding up a photo, this system requires active verification via **facial gestures**.

It uses a modular architecture where the core logic is separated from the execution scripts, making it scalable and easy to maintain.

## 📂 Directory Structure

The repository is organized to separate the source code from the project root files.

```text
face_recognition/                # Repository Root
├── face_recognition_system/     # Main Codebase
│   ├── main.py                  # Entry Point (CLI Controller)
│   ├── data/                    # Local Storage (GitIgnored)
│   │   ├── raw/                 # Captured User Images
│   │   └── models/              # Trained encodings.pickle
│   └── src/                     # Source Package
│       ├── __init__.py
│       ├── config.py            # Central Configuration
│       ├── data_loader.py       # Dataset Creation Logic
│       ├── encoder.py           # Model Training Logic
│       ├── recognition.py       # Real-time Core System
│       └── utils.py             # File System Helpers
├── .gitignore                   # Privacy protection
├── requirements.txt             # Dependencies
├── UNLICENSE                    # Public Domain License
└── README.md                    # Documentation
```

## ✨ Key Features

### 1. Modular Architecture

Code is organized into a src package. Logic, configuration, and execution are decoupled, making the codebase clean and professional.

### 2. Multi-Factor Liveness Detection

To mark a user as "Live" (Green Box), they must perform one of the following actions:

- Blink: Detected via Eye Aspect Ratio (EAR).
- Head Movement: Detected by tracking the Euclidean distance of the nose tip vector across frames.

### 3. Floating UI Design

Instead of obstructive solid boxes, the system uses a modern "floating text" design with high-contrast outlines (strokes), ensuring readability on both dark and light backgrounds.

### 4. Smart State Latching

Once a liveness action (Blink/Move) is detected, the system "remembers" the live status for a configurable duration (default: 2 seconds). This prevents the UI from flickering between "Live" and "Spoof" while the user is interacting normally.

---

## 🧠 Technical Logic: How It Works

### A. Face Encoding

We use the HOG (Histogram of Oriented Gradients) model via dlib to detect faces. The face is then transformed into a 128-dimensional vector (embedding). When identifying a user, we calculate the Euclidean distance between the live face vector and our known database.

### B. Blink Detection (EAR)

We map 6 landmarks per eye. The Eye Aspect Ratio (EAR) is calculated using the formula:

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)

- If the EAR falls below 0.24 (eyes closed) for 2 consecutive frames, a blink is registered.

### C. Movement Detection (Nose Tracking)

We track the Nose Tip landmark (Index 30 in the 68-point model).

- The system calculates the Euclidean distance of the nose tip between the current frame and the previous frame.
- If the distance > 2.5 pixels, it registers as intentional movement (nodding/turning).

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HrushiSanap/face_recognition.git
cd face_recognition
```

### 2. Install Dependencies

Note: This project requires dlib, which compiles C++ code. You may need CMake installed (pip install cmake).

```bash
pip install -r requirements.txt
```

If you face errors installing dlib on Windows, install Visual Studio C++ Build Tools first.

---

## 🛠 Usage Guide

All commands are run using the central main.py controller located inside the face_recognition_system folder.

First, navigate to the code folder:

```bash
cd face_recognition_system
```

### Step 1: Capture User Data

This will open your webcam. Look at the camera and slowly rotate your head to capture different angles.

```bash
python main.py capture --name "YourName"
```

- Controls: Press q to quit early.
- Output: Images are saved in data/raw/YourName/.

### Step 2: Train the Model

This processes the images and creates the facial embeddings.

```bash
python main.py train
```

- Output: Creates data/models/encodings.pickle.

### Step 3: Start Recognition

Run the main security system.

```bash
python main.py run
```

- Visual Indicators:
  - 🔴 Red: Unknown Face.
  - 🟡 Yellow/Cyan: Known Face, but static (Potential Spoof).
  - 🟢 Green: Known Face + Live (Blink/Move verified).
- Controls: Press q to exit.

---

## ⚙️ Configuration

You can fine-tune the system sensitivities in src/config.py.

| Parameter | Default | Description |
|---------|---------|-------------|
| EYE_AR_THRESH | 0.24 | Threshold for "eyes closed". Lower if it detects blinks too easily. |
| EYE_AR_CONSEC_FRAMES | 2 | How many frames eyes must be closed to count as a blink. |
| MOVEMENT_THRESHOLD | 2.5 | Minimum pixel distance nose must move to trigger "Live". |
| LIVENESS_TIMEOUT | 2.0 | Seconds the "Live" status persists after a gesture. |
| RECOGNITION_INTERVAL | 5 | Runs heavy face recognition every N frames to save CPU. |

---

## ❓ Troubleshooting

### 1. ModuleNotFoundError: No module named 'src'

- Ensure you are running the command from inside the face_recognition_system directory, not the root repo folder.

### 2. dlib installation fails

- Windows: Install "Desktop development with C++" workload via Visual Studio Build Tools.
- Linux: Run sudo apt-get install build-essential cmake.

### 3. System is lagging

- Open src/config.py and increase FRAME_RESIZE_SCALE (e.g., make it smaller, like 0.20) or increase RECOGNITION_INTERVAL to 10.

---

## 📜 License

This project is released under the The Unlicense.  
This means the code is dedicated to the public domain. You are free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.  
For more information, please refer to http://unlicense.org/
