# 🛡️ SafeDrive – Real-Time Driver Drowsiness Detection using FaceMesh, OpenCV & Audio Alerts

SafeDrive is a real-time drowsiness detection system using computer vision and audio alerts. It uses **MediaPipe FaceMesh** and **CVZone** to monitor a driver's eye movements via webcam, calculating the Eye Aspect Ratio (EAR). If drowsiness is detected, an alarm sound and a voice alert are triggered.

## 📌 Features

- 👁️ Real-time eye tracking using MediaPipe or CVZone
- 🔊 Audio alert using `pygame`
- 🗣️ Voice warning using `pyttsx3`
- 📊 Eye Aspect Ratio (EAR) smoothing
- 🎯 Accurate drowsiness detection based on consecutive frame analysis

---

## 🧠 How it Works

1. Detects face landmarks with MediaPipe or CVZone.
2. Calculates Eye Aspect Ratio (EAR) from 6 key eye points.
3. If the EAR remains below a threshold for several frames:
   - 🚨 Red alert border
   - 📢 Audio alarm plays
   - 🗣️ Voice warning triggers: *“Wake up, you are the driver”*

---

## 🛠️ Requirements

Install the following dependencies:

```bash
pip install opencv-python cvzone numpy pygame pyttsx3 mediapipe==0.10.20 protobuf==4.25.3
