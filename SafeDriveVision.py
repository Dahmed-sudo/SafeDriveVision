import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import os
import pygame
import time
import numpy as np
import mediapipe as mp
import pyttsx3  # Nouvelle importation pour la synthèse vocale

# ========================
# Configuration
# ========================
EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold
EAR_CONSEC_FRAMES = 20  # Frames for drowsiness detection
HISTORY_SIZE = 5  # For EAR smoothing
ALARM_FILE = "alarm.wav"  # Audio alarm file

# MediaPipe Face Landmark Config
mp_face_mesh = mp.solutions.face_mesh
FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_CONTOURS

# ========================
# Initialize Systems
# ========================
# Pygame audio setup
AUDIO_ENABLED = True
try:
    pygame.mixer.init()
    if not os.path.exists(ALARM_FILE):
        print(f"Warning: Alarm file not found at {ALARM_FILE}")
        AUDIO_ENABLED = False
except Exception as e:
    print(f"Audio disabled: {str(e)}")
    AUDIO_ENABLED = False

# Initialiser le moteur de synthèse vocale
try:
    tts_engine = pyttsx3.init()
    VOICE_ENABLED = True
except Exception as e:
    print(f"Voice synthesis disabled: {str(e)}")
    VOICE_ENABLED = False

# Initialize both detectors
cvzone_detector = FaceMeshDetector(maxFaces=1)
mp_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Eye landmark indices (MediaPipe 468-point model)
LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]

# ========================
# Utility Functions
# ========================
def play_alarm():
    if AUDIO_ENABLED:
        try:
            pygame.mixer.music.load(ALARM_FILE)
            pygame.mixer.music.play(loops=-1)
        except Exception as e:
            print(f"Audio play error: {str(e)}")

def stop_alarm():
    if AUDIO_ENABLED and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def speak_wakeup():
    if VOICE_ENABLED:
        try:
            tts_engine.say("Wake up, you are the driver")
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Voice synthesis error: {str(e)}")

def eye_aspect_ratio(eye_points):
    """Calculate normalized eye aspect ratio (EAR)"""
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def draw_landmarks(image, landmarks, connections=None):
    """Draw face landmarks on image"""
    if landmarks:
        for landmark in landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        if connections:
            for connection in connections:
                start = connection[0]
                end = connection[1]
                x1 = int(landmarks[start].x * image.shape[1])
                y1 = int(landmarks[start].y * image.shape[0])
                x2 = int(landmarks[end].x * image.shape[1])
                y2 = int(landmarks[end].y * image.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

# ========================
# Main Processing Loop
# ========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    ear_history = []
    closed_eyes_counter = 0
    alarm_active = False
    alarm_start_time = 0
    last_wakeup_time = 0  # Pour limiter la fréquence des messages vocaux

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with both detectors
        img, cvzone_faces = cvzone_detector.findFaceMesh(img, draw=False)
        mp_results = mp_detector.process(img_rgb)
        
        # Use MediaPipe results if available (more accurate)
        if mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0].landmark
            
            # Extract eye landmarks
            left_eye = np.array([(face_landmarks[idx].x * img.shape[1], 
                                 face_landmarks[idx].y * img.shape[0]) 
                             for idx in LEFT_EYE_POINTS])
            right_eye = np.array([(face_landmarks[idx].x * img.shape[1], 
                                  face_landmarks[idx].y * img.shape[0]) 
                              for idx in RIGHT_EYE_POINTS])
            
            # Draw all face landmarks
            draw_landmarks(img, face_landmarks, FACE_CONNECTIONS)
            
        # Fallback to CVZone if MediaPipe fails
        elif cvzone_faces:
            face = cvzone_faces[0]
            left_eye = np.array([face[idx] for idx in LEFT_EYE_POINTS])
            right_eye = np.array([face[idx] for idx in RIGHT_EYE_POINTS])
            
            # Draw eye landmarks only
            for point in left_eye:
                cv2.circle(img, tuple(point), 2, (255, 0, 255), -1)
            for point in right_eye:
                cv2.circle(img, tuple(point), 2, (255, 0, 255), -1)
        else:
            cvzone.putTextRect(img, "No Face Detected", (20, 50))
            cv2.imshow("Drowsiness Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Smooth EAR value
        ear_history.append(ear)
        if len(ear_history) > HISTORY_SIZE:
            ear_history.pop(0)
        smoothed_ear = sum(ear_history) / len(ear_history)
        
        # Drowsiness detection logic
        if smoothed_ear < EAR_THRESHOLD:
            closed_eyes_counter += 1
            
            if closed_eyes_counter >= EAR_CONSEC_FRAMES and not alarm_active:
                play_alarm()
                alarm_active = True
                alarm_start_time = time.time()
                status_text = "DROWSINESS DETECTED!"
                status_color = (0, 0, 255)  # Red
                
                # Activer le message vocal (max une fois toutes les 10 secondes)
                current_time = time.time()
                if current_time - last_wakeup_time > 10:
                    speak_wakeup()
                    last_wakeup_time = current_time
        else:
            closed_eyes_counter = 0
            if alarm_active:
                stop_alarm()
                alarm_active = False
            status_text = "Active (Eyes Open)"
            status_color = (0, 255, 0)  # Green
        
        # Display status
        cvzone.putTextRect(
            img, status_text, (20, 50),
            colorR=status_color,
            scale=1.5, thickness=2
        )
        cvzone.putTextRect(
            img, f"EAR: {smoothed_ear:.3f}", (20, 100),
            colorR=(255, 255, 255),
            scale=1, thickness=1
        )
        
        # Visual alarm indicator
        if alarm_active:
            cv2.rectangle(
                img, (0, 0), 
                (img.shape[1], img.shape[0]),
                (0, 0, 255), 10  # Red border
            )
        
        cv2.imshow("Drowsiness Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    mp_detector.close()

if __name__ == "__main__":
    main()