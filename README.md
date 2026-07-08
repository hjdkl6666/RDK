# Face Recognition Attack & Defense System

Real-time face recognition system with servo and LED interaction, designed for the Hobot (Horizon Robotics) embedded platform.

## Overview

This project implements an intelligent face recognition system that continuously monitors camera input. When a target face ("goal") is detected, it triggers coordinated physical actions: a servo motor rotates to simulate a physical response, and an LED strobes as a visual alert.

## File Structure

| File | Description |
|------|-------------|
| `attack.py` | Main program: real-time recognition + servo/LED coordination |
| `build_face_db_yunet.py` | Build the face feature database from image directories |
| `face_db.npy` | Generated face feature database |
| `yunet.onnx` | YuNet face detection model (ONNX format) |
| `w600k_r50.bin` | W600K-R50 face recognition model (Horizon BPU format) |

## Hardware Requirements

- **Platform**: Hobot (Horizon Robotics) embedded board (e.g., X3, X5 series)
- **Camera**: MIPI camera via `libsrcampy` (1920x1080 @ 30fps)
- **Servo Motor**: Connected to PWM channel 1 (physical pin 33)
- **LED**: Connected to GPIO physical pin 37 (GPIO26)

## Prerequisites

- Hobot DNN SDK (`hobot_dnn`)
- Hobot VIO SDK (`hobot_vio`)
- Hobot GPIO SDK
- OpenCV with `FaceDetectorYN` support (opencv-contrib)
- NumPy

## Quick Start

### 1. Build Face Database

First, prepare your face images and build the feature database.

Place target person images in `/root/fly/ai/goal/` and other person images in `/root/fly/ai/friends/`.

```bash
python build_face_db_yunet.py
```

This generates `face_db.npy` containing extracted face features.

### 2. Run the Main Program

```bash
python attack.py
```

The system will continuously:
- Capture frames from camera (1920x1080, sampled every 2 seconds)
- Detect faces using YuNet
- Extract features using W600K-R50 model
- Compare against the goal database (cosine similarity, threshold 0.5)
- If a goal face is detected and cooldown has elapsed (3s), trigger actions:
  - **Servo**: Rotates from 0° to 90° and back (hardware max speed)
  - **LED**: Strobes for 2 seconds (100ms on/off interval)

Press **Ctrl+C** to stop.

## Configuration

All configurable parameters are defined at the top of `attack.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DET_MODEL_PATH` | `/userdata/yunet.onnx` | Face detection model path |
| `REC_MODEL_PATH` | `/userdata/w600k_r50.bin` | Face recognition model path |
| `DB_PATH` | `/root/fly/ai/face_db.npy` | Feature database path |
| `CAM_WIDTH` | 1920 | Camera capture width |
| `CAM_HEIGHT` | 1080 | Camera capture height |
| `INTERVAL` | 2 | Detection interval (seconds) |
| `THRESHOLD` | 0.5 | Cosine similarity threshold |
| `SERVO_ROTATE_ANGLE` | 90 | Servo rotation angle on trigger |
| `SERVO_RESPONSE_TIME` | 0.15 | Servo wait time (seconds) |
| `BLINK_DURATION` | 2 | LED strobe duration (seconds) |
| `BLINK_INTERVAL` | 0.1 | LED on/off interval (seconds) |
| `ACTION_LOCK_TIME` | 3 | Cooldown between actions (seconds) |

## How It Works

```
Camera → YuNet Detection → Face Crop (112x112) → W600K Feature Extraction
                                                         ↓
                                              Cosine Similarity vs DB
                                                         ↓
                                             threshold > 0.5? → Servo + LED
```

### Detection Pipeline

1. **Capture**: RGB frame obtained from MIPI camera via `libsrcampy`
2. **Detection**: YuNet ONNX model detects face bounding boxes
3. **Feature Extraction**: Face regions resized to 112×112, normalized, and fed through W600K-R50 to produce a feature vector
4. **Recognition**: Cosine similarity between captured feature and goal database entries; labels as "goal" or "unknown"
5. **Action**: On positive detection (outside cooldown), both servo and LED are triggered simultaneously in separate threads to avoid blocking the main loop

## License

This project is for educational and experimental use only.
