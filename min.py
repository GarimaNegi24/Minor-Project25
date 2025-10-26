import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
import pygame   # for alarm sound
import time

# -----------------------
# Function to calculate Eye Aspect Ratio (EAR)
# -----------------------
def eye_aspect_ratio(eye):
    # vertical distances
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # horizontal distance
    C = distance.euclidean(eye[0], eye[3])
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# -----------------------
# Initialize Alarm
# -----------------------
pygame.mixer.init()
alarm_sound = "alarm.wav"  # put an alarm sound file (wav) in same folder
def sound_alarm():
    pygame.mixer.music.load(alarm_sound)
    pygame.mixer.music.play()

# -----------------------
# Constants
# -----------------------
EYE_AR_THRESH = 0.25    # EAR below which eyes are considered closed
EYE_AR_CONSEC_FRAMES = 48  # Number of consecutive frames for drowsy

COUNTER = 0
ALARM_ON = False

# -----------------------
# Load Dlib’s face detector and shape predictor
# -----------------------
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Download this file from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Grab indexes of left/right eye from the facial landmark predictor
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# -----------------------
# Start Video Stream
# -----------------------
print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Get left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        # Check if EAR is below threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    sound_alarm()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False
            pygame.mixer.music.stop()

        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow("Drowsiness Detector", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()