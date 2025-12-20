import cv2
import time
import math
from flask import Flask, render_template, Response
import mediapipe as mp

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

eyes_closed_start = None
ALERT_TIME = 2.0


def eye_aspect_ratio(landmarks, eye_ids, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_ids]
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    h1 = math.dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h1)


def gen_frames():
    global eyes_closed_start

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            ear = (
                eye_aspect_ratio(landmarks, LEFT_EYE, w, h) +
                eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ) / 2

            if ear < 0.2:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()

                elapsed = time.time() - eyes_closed_start
                cv2.putText(frame, f"DROWSY {elapsed:.1f}s",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

                if elapsed > ALERT_TIME:
                    cv2.putText(frame, "ALERT SENT!",
                                (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)
            else:
                eyes_closed_start = None

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
