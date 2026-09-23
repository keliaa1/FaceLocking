# Face Tracker & Expression Recognition

## Purpose
This project is a real-time Face Tracker that uses a camera to search for a specific person, lock onto their face, and track them dynamically across the camera's field of view. Additionally, it tracks facial expressions (such as smiling and blinking) in real time and overlays this information on a live video feed.

## Features
- **Face Enrollment**: Capture images of a user to create a facial embedding template using ArcFace ONNX.
- **Real-Time Tracking & Locking**: Fast face detection using Haar Cascades, followed by ArcFace embedding extraction to match against the database and lock onto the target face.
- **Expression Detection**: Uses MediaPipe FaceMesh to detect eye blinks (by calculating Eye Aspect Ratio) and smiles (by calculating mouth width and corner lift).
- **HUD Interface**: Real-time display of FPS, tracking lock status, expression state, and blink counts.

## Project Structure
- `main_tracker.py`: The primary tracking script. Handles video capture, face detection, expression monitoring, and HUD rendering.
- `src/enroll.py`: Script to enroll a new face into the database by capturing crops and computing an embedding.
- `src/`: Contains various components for the pipeline such as alignment, evaluation, and recognition logic.
- `models/`: Expected to store the `embedder_arcface.onnx` model and `haarcascade_frontalface_default.xml`.
- `data/db/`: Directory where the `face_db.npz` (facial embeddings database) is generated and stored upon enrollment.

## Requirements
- **Hardware**: A standard webcam connected to the host PC.
- **Software**: Python 3 with the required dependencies.

## Setup and Running

### 1. Software Setup
1. Setup a virtual environment (`.venv`) if you haven't already.
2. Install dependencies:
   ```bash
   pip install opencv-python numpy onnxruntime mediapipe requests
   ```
3. Ensure the `models/` directory contains the required `embedder_arcface.onnx` and `haarcascade_frontalface_default.xml` models.

### 2. Configuration
Open `main_tracker.py` and update the following configuration variable at the top of the file:
- `TARGET_NAME`: The name you will use for enrollment (default is "Kelia").

### 3. Enroll a Face
Before the tracker can recognize you, you must enroll your face into the database.
1. Run the enrollment script:
   ```bash
   python -m src.enroll
   ```
2. Enter your name (must match `TARGET_NAME` from configuration).
3. Follow the on-screen instructions:
   - Face the camera in stable lighting.
   - Use `a` for auto-capture, or `SPACE` for manual capture.
   - Move your head slightly to capture different angles and expressions.
   - Press `s` to save the enrollment template once enough samples are captured (around 15 are recommended).

### 4. Run the Tracker
Once enrolled, start the main tracking script:
```bash
python main_tracker.py
```
- The camera feed will appear.
- The system will scan for faces and, once detected, lock onto the enrolled face.
- It will continuously track the face and display real-time expressions (like smiling or eyes closed) and blink counts on the HUD.
- Press `q` to quit the application.
