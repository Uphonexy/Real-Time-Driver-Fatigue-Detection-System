# Real Time: Driver Fatigue Detection System

## Overview

This Python-based drowsiness detection system utilizes facial landmarks to monitor eye closure, yawning, and head pose. The system triggers alarms for enhanced safety, providing real-time alerts to users.

## Features

- **Eye Closure Detection**: Real-time monitoring of Eye Aspect Ratio (EAR)
- **Yawning Detection**: Real-time monitoring of Mouth Aspect Ratio (MAR)
- **Head Pose & Distraction Monitoring**: Detects when the driver's head is pitched down or turned away.
- **Personalized Calibration Phase**: A 5-minute initialization phase to establish a unique baseline for every driver.
- **Adaptive Thresholds**: Intelligently adjusts sensitivity based on the driver's age group and total drive duration.
- **Session Logging**: Automatically saves a detailed JSON report and prints a summary of all fatigue events when the drive concludes.
- **Distinct Audio Alerts**: Different alarm sounds for drowsiness, yawning, and distractions.

## Dependencies

- OpenCV
- dlib
- NumPy
- imutils
- Pygame

## Usage

1. Clone the repository.
2. Install the required dependencies (`pip install -r requirements.txt`).
3. Download the pre-trained facial landmark predictor file: `shape_predictor_68_face_landmarks.dat` and place it in the project root.
4. Run the main application script:
   ```bash
   python main.py
   ```
5. Follow the on-screen prompt to select your age group and begin driving normally during the 5-minute calibration phase.

## Project Structure

- `main.py`: The entry point and main video loop.
- `calibration.py`: Handles the initial baseline configuration.
- `thresholds.py`: Computes the adaptive thresholds for EAR and MAR.
- `detector.py`: Contains facial landmark math and head pose estimation.
- `alarms.py`: Manages audio alerts and cooldown logic.
- `logger.py`: Tracks fatigue events and generates the JSON session summary.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



