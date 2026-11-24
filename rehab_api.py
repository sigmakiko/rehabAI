
import os
import cv2
import tempfile
import numpy as np
import mediapipe as mp

from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="SporGo Rehab AI API")

mp_pose = mp.solutions.pose

# Auto-detect both sides: elbows, knees, shoulders and hips
JOINTS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
}

def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return None

    cosine = np.dot(ba, bc) / denominator
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    return float(angle)


@app.get("/health")
def health():
    return {"status": "running", "service": "SporGo Rehab AI"}


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    video_path = None
    cap = None
    max_size = 50 * 1024 * 1024
    total_size = 0

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp:
            video_path = temp.name

            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="Video is larger than 50MB"
                    )
                temp.write(chunk)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or not np.isfinite(fps):
            fps = 30.0

        frame_count = 0
        sample_step = max(1, round(fps / 10))
        angles = {name: [] for name in JOINTS}

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_count % sample_step != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if not result.pose_landmarks:
                    continue

                landmarks = result.pose_landmarks.landmark
                h, w = frame.shape[:2]

                for name, ids in JOINTS.items():
                    points = [landmarks[i] for i in ids]

                    if min(p.visibility for p in points) < 0.6:
                        continue

                    if not all(
                        0 <= p.x <= 1 and 0 <= p.y <= 1
                        for p in points
                    ):
                        continue

                    coords = [(p.x * w, p.y * h) for p in points]
                    angle = calculate_angle(*coords)

                    if angle is not None:
                        angles[name].append(angle)

        summary = {}
        for name, values in angles.items():
            if values:
                summary[name] = {
                    "min_angle": round(min(values), 1),
                    "max_angle": round(max(values), 1),
                    "samples": len(values),
                }

        return {
            "status": "success",
            "duration_seconds": round(frame_count / fps, 2),
            "joints": summary,
        }

    finally:
        if cap is not None:
            cap.release()
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        await file.close()
