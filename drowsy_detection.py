"""
This script is used to detect drowsiness and yawning
"""

import cv2
import mediapipe as mp
from scipy.spatial import distance as dist

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 48
MOUTH_AR_THRESH = 0.75
MOUTH_AR_CONSEC_FRAMES = 20

LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_IDXS = [13, 14, 61, 291]


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[1])
    B = dist.euclidean(mouth[2], mouth[3])
    return A / B


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

EYE_COUNTER = 0
MOUTH_COUNTER = 0
DROWSY = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_eye_coords = [
                (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
                for i in LEFT_EYE_IDXS
            ]
            right_eye_coords = [
                (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
                for i in RIGHT_EYE_IDXS
            ]
            mouth_coords = [
                (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
                for i in MOUTH_IDXS
            ]

            for point in left_eye_coords:
                cv2.circle(image, point, 2, (0, 255, 0), -1)  # Green circle
            for point in right_eye_coords:
                cv2.circle(image, point, 2, (0, 255, 0), -1)
            for point in mouth_coords:
                cv2.circle(image, point, 2, (0, 255, 0), -1)

            left_ear = eye_aspect_ratio(left_eye_coords)
            right_ear = eye_aspect_ratio(right_eye_coords)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth_coords)

            cv2.putText(
                image,
                f"EAR: {ear:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"MAR: {mar:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if ear < EYE_AR_THRESH:
                EYE_COUNTER += 1
                if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    DROWSY = True
                    cv2.putText(
                        image,
                        "DROWSINESS ALERT!",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
            else:
                EYE_COUNTER = 0
                DROWSY = False

            if mar > MOUTH_AR_THRESH:
                MOUTH_COUNTER += 1
                if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    cv2.putText(
                        image,
                        "YAWN DETECTED!",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
            else:
                MOUTH_COUNTER = 0

    cv2.imshow("Drowsiness Detection", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
