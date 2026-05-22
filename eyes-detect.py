import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SOUND_PATH = os.path.join(os.path.dirname(__file__),
                          "freesound_community-siren-alert-96052.mp3")

alert_proc = None   # single tracked afplay process


def play_alert():
    """Start alert sound; kill any already-playing instance first."""
    global alert_proc
    stop_alert()
    alert_proc = subprocess.Popen(["afplay", SOUND_PATH],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)


def stop_alert():
    """Stop the alert sound if it is playing."""
    global alert_proc
    if alert_proc is not None and alert_proc.poll() is None:
        alert_proc.terminate()
    alert_proc = None

# ── Download face landmarker model once (5.8 MB) ──────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~5.8 MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH,
    )
    print("Download complete.")

# ── Eye landmark indices (MediaPipe 468-point mesh) ───────────────────────────
# Order per eye: [outer, upper1, upper2, inner, lower2, lower1]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.22          # below → closed


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)


# ── Build FaceLandmarker ──────────────────────────────────────────────────────
options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
eyes_open = True   # track previous state to trigger sound on transition

with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_img)

        label = "No Face Detected"
        color = (128, 128, 128)
        currently_open = True   # default assumption

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            avg_ear   = (right_ear + left_ear) / 2.0

            currently_open = avg_ear >= EAR_THRESHOLD

            if currently_open:
                label = f"Eyes OPEN  (EAR: {avg_ear:.2f})"
                color = (0, 255, 0)    # green
                # Stop sound when eyes open
                if not eyes_open:
                    stop_alert()
            else:
                label = f"Eyes CLOSED (EAR: {avg_ear:.2f})"
                color = (0, 0, 255)    # red
                # Play alert only on the transition open → closed
                if eyes_open:
                    play_alert()

            eyes_open = currently_open

            for idx in RIGHT_EYE + LEFT_EYE:
                x = int(lm[idx].x * w)
                y = int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 3, color, -1)

        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.putText(frame, "Press 'q' to quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

stop_alert()
cap.release()
cv2.destroyAllWindows()