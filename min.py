import cv2
import pygame   # for alarm sound
import time
import mediapipe as mp
from scipy.spatial import distance as dist

# ------------- CONFIGURATIONS -------------
EAR_THRESH = 0.23          # Eye closed threshold (tune for your face)
CLOSED_SECONDS = 2.0       # Trigger alarm if eyes closed this long
CAM_INDEX = 0              # Default webcam

# Selected Face Mesh eye landmark indices (left & right)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ------------- INITIALIZE ALARM (pygame) -------------
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")
except pygame.error:
    alarm_sound = None
    print("Warning: alarm.wav not found or cannot be loaded. Alarm will be silent.")

def play_alarm():
    if alarm_sound is not None:
        if not pygame.mixer.get_busy():
            alarm_sound.play(-1)  # loop alarm
def stop_alarm():
    if alarm_sound is not None:
        alarm_sound.stop()

# ------------- EAR CALCULATION -------------
def eye_aspect_ratio(eye_points):
    # eye_points: list of (x, y) coordinates
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ------------- MEDIAPIPE SETUP -------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ------------- MAIN LOOP -------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    closed_start_time = None
    alarm_on = False

    cv2.namedWindow("Road Accident Prevention - Eye Blink Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip and convert BGR to RGB for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)
        h, w = frame.shape[:2]

        status_text = "No Face"
        ear_value = 0.0

        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0]
            # convert all landmarks to (x, y)
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in mesh.landmark]

            left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
            right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]

            # draw eye landmarks
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear_value = (left_ear + right_ear) / 2.0

            # Determine eye status
            if ear_value < EAR_THRESH:
                status_text = "Eyes Closed"
                if closed_start_time is None:
                    closed_start_time = time.time()
                closed_duration = time.time() - closed_start_time

                # Trigger alarm if closed for too long
                if closed_duration >= CLOSED_SECONDS and not alarm_on:
                    play_alarm()
                    alarm_on = True
            else:
                status_text = "Eyes Open"
                closed_start_time = None
                if alarm_on:
                    stop_alarm()
                    alarm_on = False
        else:
            # No face detected: reset closure timer and stop alarm
            closed_start_time = None
            if alarm_on:
                stop_alarm()
                alarm_on = False

        # Draw EAR and status
        ear_display = f"EAR: {ear_value:.3f}"
        cv2.putText(frame, ear_display, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (72, 255, 236), 2)

        cv2.putText(frame, f"Status: {status_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if status_text == "Eyes Open" else (0, 0, 255),
                    2)

        alarm_status = "Alarm: ON" if alarm_on else "Alarm: OFF"
        cv2.putText(frame, alarm_status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if alarm_on else (255, 255, 255),
                    2)

        cv2.putText(frame, "Press 'q' to quit", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Road Accident Prevention - Eye Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()