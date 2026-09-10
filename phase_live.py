"""
Phase Live: Live Presentation Coach & Live Analyzer
Role: Handles real-time video/audio WebSocket streams and compiles the final scorecard with historical comparisons.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import os
import json
import base64
import math
import re
import tempfile
from datetime import datetime, timezone
import google.generativeai as genai
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import PresentationSession, HistoricalReport, db
from groq import Groq

# ===== GRACEFUL NATIVE LIBRARY IMPORTS =====
# MediaPipe, OpenCV, Librosa, and Soundfile can have complex native dependencies.
# We import them inside try-except blocks so the server starts correctly on any host.

OPENCV_AVAILABLE = False
OPENCV_IMPORT_ERROR = None
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    OPENCV_AVAILABLE = True
except Exception as e:
    OPENCV_IMPORT_ERROR = str(e)
    print(f"[LIVE WARN] OpenCV or NumPy not available: {OPENCV_IMPORT_ERROR}")

MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_IMPORTED = False
MEDIAPIPE_VERSION = None
MEDIAPIPE_IMPORT_ERROR = None
MEDIAPIPE_MODE = None  # "solutions" or "tasks"

try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_IMPORTED = True
    MEDIAPIPE_VERSION = getattr(mp, "__version__", None)

    # 1. Try legacy Solutions API (older MediaPipe)
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = getattr(mp.solutions, "pose", None)
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_MODE = "solutions"
    else:
        # 2. Modern MediaPipe Tasks API (MediaPipe >= 0.10.14 / 1.0+)
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_dir = os.path.dirname(os.path.abspath(__file__))
            task_paths = [
                os.path.join(base_dir, "cascades", "face_landmarker.task"),
                os.path.join(base_dir, "downloads", "face_landmarker.task"),
            ]
            task_model_path = next((p for p in task_paths if os.path.exists(p)), None)

            if task_model_path:
                MEDIAPIPE_AVAILABLE = True
                MEDIAPIPE_MODE = "tasks"
                MEDIAPIPE_IMPORT_ERROR = None
            else:
                MEDIAPIPE_IMPORT_ERROR = "MediaPipe face_landmarker.task model file not found."
        except Exception as task_err:
            MEDIAPIPE_IMPORT_ERROR = f"MediaPipe Tasks API unavailable: {str(task_err)}"
except Exception as e:
    MEDIAPIPE_IMPORT_ERROR = str(e)
    print(f"[LIVE WARN] MediaPipe not available: {MEDIAPIPE_IMPORT_ERROR}")

LIBROSA_AVAILABLE = False
try:
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore
    LIBROSA_AVAILABLE = True
except ImportError:
    print("[LIVE WARN] Librosa or Soundfile not available. Using mock voice feature extraction.")


# Initialize Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_available = False
if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key-here':
    try:
        genai.configure(api_key=GEMINI_API_KEY, transport='rest')
        gemini_available = True
    except Exception as e:
        print(f"[LIVE WARN] Failed to configure Gemini API: {str(e)}")

# Initialize Groq client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != 'your-groq-api-key-here':
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("[LIVE OK] Groq API configured successfully for Live Presentation Coach")
    except Exception as e:
        print(f"[LIVE WARN] Failed to configure Groq client: {str(e)}")

# Create Blueprint
phase_live_bp = Blueprint('phase_live', __name__, url_prefix='/api/presentation')

# ===== SCORING RELIABILITY THRESHOLDS (FIX) =====
# Minimum required samples to compute stable visual and audio averages
MIN_VALID_VIDEO_SAMPLES = 3     # need at least 3 valid frames to trust visual presence
MIN_VALID_AUDIO_SAMPLES = 1     # need at least 1 audio chunk to measure vocal delivery
MIN_WORDS_PER_CHUNK = 2         # Whisper hallucinates 1-word phrases on silence/noise

# ===== REAL-TIME FEATURE EXTRACTION FUNCTIONS =====

face_cascade = None
face_cascade_alt2 = None
profile_cascade = None
eye_cascade = None
face_mesh_detector = None

def init_cascades():
    global face_cascade, face_cascade_alt2, profile_cascade, eye_cascade
    if OPENCV_AVAILABLE and (face_cascade is None or face_cascade_alt2 is None):
        try:
            local_cascades = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades")
            cascade_dir = getattr(cv2.data, "haarcascades", "")

            def get_path(filename):
                local = os.path.join(local_cascades, filename)
                if os.path.exists(local):
                    return local
                system = os.path.join(cascade_dir, filename)
                if os.path.exists(system):
                    return system
                return None

            alt2_path = get_path("haarcascade_frontalface_alt2.xml")
            default_path = get_path("haarcascade_frontalface_default.xml")
            profile_path = get_path("haarcascade_profileface.xml")
            eye_path = get_path("haarcascade_eye.xml")

            if alt2_path:
                face_cascade_alt2 = cv2.CascadeClassifier(alt2_path)
            if default_path:
                face_cascade = cv2.CascadeClassifier(default_path)
            if profile_path:
                profile_cascade = cv2.CascadeClassifier(profile_path)
            if eye_path:
                eye_cascade = cv2.CascadeClassifier(eye_path)

            print(f"[LIVE OK] Haar Cascades loaded: alt2={face_cascade_alt2 is not None}, default={face_cascade is not None}, eyes={eye_cascade is not None}")
        except Exception as e:
            print(f"[LIVE WARN] Failed to load OpenCV cascades: {str(e)}")

def init_face_mesh():
    global face_mesh_detector
    if not MEDIAPIPE_AVAILABLE or face_mesh_detector is not None:
        return

    if MEDIAPIPE_MODE == "solutions":
        try:
            face_mesh_detector = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35
            )
            print("[LIVE OK] MediaPipe Solutions FaceMesh initialized!")
        except Exception as e:
            print(f"[LIVE WARN] Failed to init MediaPipe solutions FaceMesh: {e}")
    elif MEDIAPIPE_MODE == "tasks":
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_dir = os.path.dirname(os.path.abspath(__file__))
            task_paths = [
                os.path.join(base_dir, "cascades", "face_landmarker.task"),
                os.path.join(base_dir, "downloads", "face_landmarker.task"),
            ]
            task_model_path = next((p for p in task_paths if os.path.exists(p)), None)
            if task_model_path:
                base_options = mp_python.BaseOptions(model_asset_path=task_model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=0.25,
                    min_face_presence_confidence=0.25,
                    min_tracking_confidence=0.25
                )
                face_mesh_detector = mp_vision.FaceLandmarker.create_from_options(options)
                print("[LIVE OK] MediaPipe Tasks FaceLandmarker initialized successfully!")
        except Exception as e:
            print(f"[LIVE WARN] Failed to init MediaPipe Tasks FaceLandmarker: {e}")

def _clip_score(value, low=0, high=100):
    return int(max(low, min(high, value)))


def _calculate_score_stability(values):
    if not values or len(values) < 2:
        return 1.0
    mean_value = sum(values) / len(values)
    if mean_value <= 0:
        return 0.0
    variance = sum((x - mean_value) ** 2 for x in values) / len(values)
    stddev = math.sqrt(variance)
    return max(0.0, 1.0 - min(1.0, stddev / mean_value))


def _calculate_confidence(eye_contact_score, posture_score, visibility_score=1.0, session=None):
    if eye_contact_score <= 0 or posture_score <= 0 or visibility_score <= 0:
        return 0

    quality = math.sqrt((eye_contact_score / 100.0) * (posture_score / 100.0))
    stability = 1.0
    if session is not None:
        eye_stability = _calculate_score_stability(session.metrics.get("eye_contact_scores", [])[-8:])
        posture_stability = _calculate_score_stability(session.metrics.get("posture_scores", [])[-8:])
        stability = (eye_stability + posture_stability) / 2.0

    confidence = quality * stability * visibility_score
    return _clip_score(confidence * 100, 0, 100)


def _smooth_visual_scores(eye_contact_score, posture_score, session, face_detected=True):
    if not face_detected or session is None:
        return eye_contact_score, posture_score

    prev_eyes = session.metrics.get("eye_contact_scores", [])[-4:]
    prev_postures = session.metrics.get("posture_scores", [])[-4:]

    # FIX: If eye contact is zero or low (looking away), drop immediately to zero/low!
    # Do NOT smooth low scores with past high scores, otherwise zero eye contact is masked.
    if eye_contact_score < 35:
        smoothed_eye = eye_contact_score
    elif prev_eyes:
        avg_prev_eye = sum(prev_eyes) / len(prev_eyes)
        if eye_contact_score < avg_prev_eye:
            smoothed_eye = int(0.90 * eye_contact_score + 0.10 * avg_prev_eye)
        else:
            smoothed_eye = int(0.50 * eye_contact_score + 0.50 * avg_prev_eye)
    else:
        smoothed_eye = eye_contact_score

    if posture_score < 35:
        smoothed_posture = posture_score
    elif prev_postures:
        avg_prev_posture = sum(prev_postures) / len(prev_postures)
        if posture_score < avg_prev_posture:
            smoothed_posture = int(0.80 * posture_score + 0.20 * avg_prev_posture)
        else:
            smoothed_posture = int(0.50 * posture_score + 0.50 * avg_prev_posture)
    else:
        smoothed_posture = posture_score

    return smoothed_eye, smoothed_posture


def _visual_result(eye_contact_score, posture_score, hint, emotion=None, session=None, face_detected=True, visibility_score=1.0):
    eye_contact_score, posture_score = _smooth_visual_scores(
        _clip_score(eye_contact_score),
        _clip_score(posture_score),
        session,
        face_detected=face_detected
    )
    confidence_score = _calculate_confidence(
        eye_contact_score,
        posture_score,
        visibility_score=visibility_score,
        session=session
    ) if face_detected else 0

    if emotion is None:
        if not face_detected:
            emotion = "NOT DETECTED"
        elif eye_contact_score >= 80 and posture_score >= 80:
            emotion = "Confident"
        elif eye_contact_score >= 65 and posture_score >= 65:
            emotion = "Focused"
        elif eye_contact_score < 45:
            emotion = "Distracted"
        else:
            emotion = "Nervous"

    return {
        "face_detected": face_detected,
        "eye_contact": eye_contact_score,
        "eye_contact_score": eye_contact_score,
        "posture": posture_score,
        "posture_score": posture_score,
        "hint": hint,
        "confidence": confidence_score,
        "confidence_score": confidence_score,
        "emotion": emotion,
        "valid": face_detected
    }


def _unmeasured_visual_result(hint):
    return {
        "face_detected": False,
        "eye_contact_score": 0,
        "posture_score": 0,
        "confidence_score": 0,
        "hint": hint,  # FIX: was missing, so the reason never reached the caller/frontend
        "emotion": "NOT DETECTED",
        "valid": False
    }

def _decode_frame(base64_image_data):
    if not OPENCV_AVAILABLE:
        return None
    if "," in base64_image_data:
        base64_image_data = base64_image_data.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def _analyze_frame_with_mediapipe(img, session=None):
    if not MEDIAPIPE_AVAILABLE:
        return None

    init_face_mesh()
    if face_mesh_detector is None:
        return None

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = None

    if MEDIAPIPE_MODE == "solutions":
        results = face_mesh_detector.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
        else:
            try:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
                cl = clahe.apply(l_channel)
                enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                res_retry = face_mesh_detector.process(enhanced_rgb)
                if res_retry.multi_face_landmarks:
                    landmarks = res_retry.multi_face_landmarks[0].landmark
            except Exception:
                pass
    elif MEDIAPIPE_MODE == "tasks":
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = face_mesh_detector.detect(mp_image)
            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
            else:
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
                    cl = clahe.apply(l_channel)
                    enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                    mp_image_retry = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_rgb)
                    res_retry = face_mesh_detector.detect(mp_image_retry)
                    if res_retry.face_landmarks:
                        landmarks = res_retry.face_landmarks[0]
                except Exception:
                    pass
        except Exception:
            landmarks = None

    if not landmarks:
        # Crucial: return None so that Haar cascade detection with CLAHE runs as fallback!
        return None

    def point(index):
        lm = landmarks[index]
        return lm.x * w, lm.y * h

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    face_cx = ((min(xs) + max(xs)) / 2) * w
    face_cy = ((min(ys) + max(ys)) / 2) * h
    face_height_ratio = max(ys) - min(ys)
    face_visibility = max(0.0, min(1.0, (face_height_ratio - 0.12) / 0.4))

    left_eye_outer = point(33)
    left_eye_inner = point(133)
    right_eye_inner = point(362)
    right_eye_outer = point(263)

    left_eye_top = point(159)
    left_eye_bottom = point(145)
    right_eye_top = point(386)
    right_eye_bottom = point(374)

    left_eye_width = max(1.0, abs(left_eye_inner[0] - left_eye_outer[0]))
    right_eye_width = max(1.0, abs(right_eye_outer[0] - right_eye_inner[0]))
    left_eye_height = max(1.0, abs(left_eye_bottom[1] - left_eye_top[1]))
    right_eye_height = max(1.0, abs(right_eye_bottom[1] - right_eye_top[1]))

    # Eye Aspect Ratio (EAR) - detect closed eyes or looking straight down
    left_ear = left_eye_height / left_eye_width
    right_ear = right_eye_height / right_eye_width
    avg_ear = (left_ear + right_ear) / 2.0

    left_iris = [point(i) for i in range(468, 473)]
    right_iris = [point(i) for i in range(473, 478)]
    left_iris_x = sum(p[0] for p in left_iris) / len(left_iris)
    right_iris_x = sum(p[0] for p in right_iris) / len(right_iris)
    left_iris_y = sum(p[1] for p in left_iris) / len(left_iris)
    right_iris_y = sum(p[1] for p in right_iris) / len(right_iris)

    # Horizontal iris ratio (0.50 = perfectly centered)
    left_h_ratio = (left_iris_x - min(left_eye_outer[0], left_eye_inner[0])) / left_eye_width
    right_h_ratio = (right_iris_x - min(right_eye_inner[0], right_eye_outer[0])) / right_eye_width
    gaze_h_dev = (abs(left_h_ratio - 0.50) + abs(right_h_ratio - 0.50)) / 2.0

    # Vertical iris ratio (0.42 = looking directly at camera lens; >0.56 = looking down at screen/notes)
    left_v_ratio = (left_iris_y - min(left_eye_top[1], left_eye_bottom[1])) / left_eye_height
    right_v_ratio = (right_iris_y - min(right_eye_top[1], right_eye_bottom[1])) / right_eye_height
    gaze_v_dev = (abs(left_v_ratio - 0.42) + abs(right_v_ratio - 0.42)) / 2.0

    # Head orientation / turn deviation (Nose 1, Chin 152)
    eye_center_x = (left_eye_outer[0] + right_eye_outer[0]) / 2.0
    eye_center_y = (left_eye_outer[1] + right_eye_outer[1]) / 2.0
    eye_span = max(1.0, abs(right_eye_outer[0] - left_eye_outer[0]))
    face_len = max(1.0, abs(point(152)[1] - point(1)[1]))

    head_yaw_dev = abs(point(1)[0] - eye_center_x) / eye_span
    head_pitch_dev = abs((point(1)[1] - eye_center_y) / face_len - 0.55)

    # Strict Eye Contact Check: If eyes are closed, looking down at screen/desk, or head turned away:
    is_looking_away = (
        avg_ear < 0.16 or
        left_v_ratio > 0.56 or right_v_ratio > 0.56 or
        gaze_v_dev > 0.14 or gaze_h_dev > 0.12 or
        head_yaw_dev > 0.14 or head_pitch_dev > 0.16
    )

    if is_looking_away:
        eye_contact_score = 0
    else:
        total_dev = (gaze_h_dev * 3.5) + (gaze_v_dev * 4.5) + (head_yaw_dev * 2.5) + (head_pitch_dev * 2.5)
        eye_contact_score = _clip_score(100 - (total_dev * 200), 0, 100)
        if total_dev > 0.28:
            eye_contact_score = max(0, eye_contact_score - 30)

    ideal_cx = w / 2
    ideal_cy = h * 0.42
    dev_x = abs(face_cx - ideal_cx) / w
    dev_y = abs(face_cy - ideal_cy) / h

    eye_dx = right_eye_outer[0] - left_eye_outer[0]
    eye_dy = right_eye_outer[1] - left_eye_outer[1]
    eye_tilt = abs(eye_dy) / max(1.0, abs(eye_dx))

    size_penalty = 0
    if face_height_ratio < 0.24:
        size_penalty = (0.24 - face_height_ratio) * 130
    elif face_height_ratio > 0.72:
        size_penalty = (face_height_ratio - 0.72) * 100

    posture_score = _clip_score(
        100 - (dev_x * 130) - (dev_y * 115) - (eye_tilt * 180) - size_penalty,
        0,
        100
    )

    if posture_score < 70:
        if dev_x > 0.15:
            hint = "Center yourself in front of the camera."
        elif face_height_ratio < 0.24:
            hint = "Move closer so your face is clearly visible."
        elif eye_tilt > 0.12:
            hint = "Keep your head level and shoulders steady."
        else:
            hint = "Sit upright and keep a steady posture."
    elif eye_contact_score < 70:
        if gaze_v_dev > 0.15:
            hint = "Look up at the camera lens instead of down at your notes."
        else:
            hint = "Look closer to the camera lens for stronger eye contact."
    else:
        hint = "Good eye contact and posture!"

    return _visual_result(
        eye_contact_score,
        posture_score,
        hint,
        session=session,
        face_detected=True,
        visibility_score=face_visibility
    )


def _analyze_frame_with_haar(img, session=None):
    init_cascades()

    if face_cascade is None or eye_cascade is None:
        return None

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE Adaptive Contrast Equalization (dramatically improves low-light webcams)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    def _safe_detect(cascade, im, sf=1.15, mn=3, ms=(30, 30)):
        if cascade is None or im is None:
            return []
        try:
            res = cascade.detectMultiScale(im, scaleFactor=max(1.10, sf), minNeighbors=mn, minSize=ms)
            return list(res) if len(res) > 0 else []
        except Exception:
            return []

    faces = _safe_detect(face_cascade_alt2, enhanced_gray, sf=1.15, mn=3, ms=(30, 30))

    if len(faces) == 0:
        faces = _safe_detect(face_cascade, enhanced_gray, sf=1.15, mn=3, ms=(30, 30))

    if len(faces) == 0:
        faces = _safe_detect(face_cascade_alt2, gray, sf=1.15, mn=3, ms=(30, 30))

    if len(faces) == 0:
        faces = _safe_detect(profile_cascade, enhanced_gray, sf=1.18, mn=3, ms=(30, 30))

    if len(faces) == 0:
        eq_gray = cv2.equalizeHist(gray)
        faces = _safe_detect(face_cascade_alt2, eq_gray, sf=1.15, mn=2, ms=(25, 25))
        if len(faces) == 0:
            faces = _safe_detect(face_cascade, eq_gray, sf=1.15, mn=2, ms=(25, 25))

    if len(faces) == 0:
        return _unmeasured_visual_result("Face not detected. Ensure adequate lighting and look at the camera.")

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_cx = fx + fw / 2
    face_cy = fy + fh / 2
    ideal_cx = w / 2
    ideal_cy = h * 0.35

    dev_x = abs(face_cx - ideal_cx) / w
    dev_y = (face_cy - ideal_cy) / h
    face_height_ratio = fh / h

    x_penalty = min(30, dev_x * 120)
    y_penalty = min(40, dev_y * 133) if dev_y > 0 else min(15, abs(dev_y) * 75)
    size_penalty = 0
    if face_height_ratio < 0.2:
        size_penalty = min(20, (0.2 - face_height_ratio) * 100)
    elif face_height_ratio > 0.65:
        size_penalty = min(15, (face_height_ratio - 0.65) * 80)

    posture_score = _clip_score(100 - x_penalty - y_penalty - size_penalty, 0, 100)

    face_roi_enhanced = enhanced_gray[fy:fy + fh, fx:fx + fw]
    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]
    eyes = _safe_detect(eye_cascade, face_roi_enhanced, sf=1.12, mn=3, ms=(10, 10))
    if len(eyes) == 0:
        eyes = _safe_detect(eye_cascade, face_roi_gray, sf=1.12, mn=2, ms=(10, 10))

    face_visibility = max(0.0, min(1.0, (fw * fh) / (w * h)))

    if len(eyes) == 0:
        # Face is clearly present! Position/posture is solid.
        # Estimate eye contact based on head centering rather than dropping to 0.
        centered_bonus = max(0, int(35 - dev_x * 100))
        eye_contact_score = int(_clip_score(45 + centered_bonus, 30, 80))
        hint = "Face detected. Look directly at the camera lens for optimal eye contact." if posture_score >= 70 else "Sit upright and look at the camera."
        return _visual_result(
            eye_contact_score,
            posture_score,
            hint,
            emotion="Focused" if posture_score >= 70 else "Nervous",
            session=session,
            face_detected=True,
            visibility_score=face_visibility
        )

    face_area = fw * fh
    eye_metrics = []
    for (ex, ey, ew, eh) in eyes:
        eye_cx = ex + (ew / 2)
        eye_cy = ey + (eh / 2)
        area_ratio = (ew * eh) / face_area
        visibility_score = max(0.0, min(1.0, (area_ratio - 0.01) / 0.04))
        horizontal_offset = abs(eye_cx - (fw * 0.5)) / fw
        vertical_offset = abs(eye_cy - (fh * 0.45)) / fh

        # Pupil dark center detection inside eye ROI
        gaze_dev_roi = 0.0
        try:
            eye_crop = face_roi_gray[ey:ey + eh, ex:ex + ew]
            if eye_crop.size > 0:
                blurred = cv2.GaussianBlur(eye_crop, (5, 5), 0)
                _, _, min_loc, _ = cv2.minMaxLoc(blurred)
                pupil_x, pupil_y = min_loc
                pupil_h_ratio = pupil_x / float(max(1, ew))
                pupil_v_ratio = pupil_y / float(max(1, eh))
                gaze_dev_roi = abs(pupil_h_ratio - 0.50) + abs(pupil_v_ratio - 0.40)
        except Exception:
            gaze_dev_roi = 0.15

        eye_metrics.append({
            "cx": eye_cx,
            "cy": eye_cy,
            "w": ew,
            "h": eh,
            "visibility": visibility_score,
            "horizontal_offset": min(1.0, horizontal_offset * 2),
            "vertical_offset": min(1.0, vertical_offset * 3),
            "gaze_dev": gaze_dev_roi
        })

    avg_gaze_dev = sum(e["gaze_dev"] for e in eye_metrics) / len(eye_metrics)
    avg_center_offset = sum(e["horizontal_offset"] for e in eye_metrics) / len(eye_metrics)

    if len(eye_metrics) == 1:
        # Single eye detected (slight head turn or shadow)
        dev_penalty = min(50, avg_gaze_dev * 200 + avg_center_offset * 30)
        eye_contact_score = int(_clip_score(70 - dev_penalty, 25, 80))
    elif avg_gaze_dev > 0.18 or avg_center_offset > 0.22:
        # User is looking significantly away from camera
        eye_contact_score = 20
    else:
        gaze_penalty = min(70, avg_gaze_dev * 180)
        left_eye, right_eye = sorted(eye_metrics[:2], key=lambda e: e["cx"])
        avg_visibility = (left_eye["visibility"] + right_eye["visibility"]) / 2
        vertical_alignment = 1.0 - min(1.0, abs(left_eye["cy"] - right_eye["cy"]) / max(1.0, fh * 0.08))
        symmetry = (
            min(left_eye["w"], right_eye["w"]) / max(1.0, max(left_eye["w"], right_eye["w"])) +
            min(left_eye["h"], right_eye["h"]) / max(1.0, max(left_eye["h"], right_eye["h"]))
        ) / 2
        eye_distance_ratio = abs(right_eye["cx"] - left_eye["cx"]) / max(1.0, fw)
        expected_distance = 0.28
        distance_alignment = 1.0 - min(1.0, abs(eye_distance_ratio - expected_distance) / expected_distance)

        base_raw_score = 100 * (
            0.35 * avg_visibility +
            0.25 * (1.0 - avg_center_offset) +
            0.20 * vertical_alignment +
            0.10 * symmetry +
            0.10 * distance_alignment
        )
        eye_contact_score = int(_clip_score(base_raw_score - gaze_penalty, 30, 100))

    if posture_score < 70:
        if dev_y > 0.15:
            hint = "Sit upright! You are slouching."
        elif dev_x > 0.15:
            hint = "Center yourself in front of the camera."
        elif face_height_ratio < 0.2:
            hint = "Move a bit closer to the camera."
        else:
            hint = "Adjust your posture to sit straight."
    elif eye_contact_score < 60:
        hint = "Look directly into the camera lens."
    else:
        hint = "Good eye contact and posture!"

    return _visual_result(
        eye_contact_score,
        posture_score,
        hint,
        session=session,
        face_detected=True,
        visibility_score=face_visibility
    )

def analyze_webcam_frame(base64_image_data: str, session=None) -> dict:
    """
    Analyzes a base64 encoded video frame for eye contact and posture alignment.
    Uses MediaPipe Face Mesh when available, with OpenCV Haar Cascades as a real fallback.
    """
    if not base64_image_data:
        return _unmeasured_visual_result("No video signal was received.")

    try:
        img = _decode_frame(base64_image_data)
        if img is None:
            reason = OPENCV_IMPORT_ERROR or "OpenCV could not decode this frame."
            return _unmeasured_visual_result(f"Camera frame could not be processed: {reason}")

        mediapipe_result = _analyze_frame_with_mediapipe(img, session)
        if mediapipe_result is not None:
            return mediapipe_result

        haar_result = _analyze_frame_with_haar(img, session)
        if haar_result is not None:
            return haar_result

        reason = MEDIAPIPE_IMPORT_ERROR or "MediaPipe is unavailable and OpenCV Haar cascades did not initialize."
        return _unmeasured_visual_result(f"Camera analysis unavailable: {reason}")
    except Exception as e:
        print(f"[LIVE ERROR] Frame processing failed: {str(e)}")
        return _unmeasured_visual_result("Camera analysis failed for this frame.")


def analyze_audio_chunk(base64_audio_data: str, session_id: str = "live", transcript_hint: str = "") -> dict:
    """
    Processes voice chunks for pacing (WPM) and filler words.
    Uses Groq Whisper API for real-time transcription if available,
    optionally uses a client-provided transcript. If neither is available,
    the chunk is marked as unmeasured instead of generating dummy values.
    """
    if not base64_audio_data:
        return {"wpm": 0, "filler_word_detected": False, "transcript": "", "filler_count": 0, "valid": False}

    try:
        # Clean header
        if "," in base64_audio_data:
            base64_audio_data = base64_audio_data.split(",")[1]
            
        audio_bytes = base64.b64decode(base64_audio_data)
        
        transcript = ""
        filler_count = 0
        filler_detected = False
        temp_audio_path = None
        
        # 1. Use Groq Whisper STT if API key is set
        if groq_client:
            try:
                # Save base64-decoded bytes to a temp file
                # WebM is the format browser MediaRecorder outputs in useLiveSession.ts
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                    temp_audio.write(audio_bytes)
                    temp_audio_path = temp_audio.name
                
                print(f"🎙️ [LIVE STT] Transcribing 3s chunk using Groq Whisper...")
                with open(temp_audio_path, "rb") as audio_file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=(f"chunk_{session_id}.webm", audio_file.read()),
                        model="whisper-large-v3",
                        language="en"
                    )
                    transcript = transcription.text.strip()
                
                # Cleanup local temp file
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    pass
                    
            except Exception as e:
                print(f"[LIVE STT WARN] Groq Whisper chunk transcription failed: {str(e)}")
                transcript = ""

        if not transcript and transcript_hint:
            transcript = transcript_hint.strip()
        
        # 2. Process real transcript if STT succeeded
        if transcript:
            print(f"🎙️ [LIVE STT RESULT] Transcript: '{transcript}'")
            words = transcript.split()
            word_count = len([w for w in words if w.strip()])

            # ── FIX: Whisper (and most Whisper-family models) is known to
            # hallucinate short filler phrases ("you", "thank you", "bye")
            # on silence or background noise. A 1-word transcript from a 3s
            # chunk is almost never real speech — treat it as unmeasured
            # instead of letting it produce a fake WPM/score.
            if word_count < MIN_WORDS_PER_CHUNK:
                print(f"[LIVE STT WARN] Discarding likely-hallucinated chunk transcript: '{transcript}' ({word_count} word(s)).")
                return {
                    "wpm": 0,
                    "filler_word_detected": False,
                    "transcript": "",
                    "filler_count": 0,
                    "pitch_score": None,
                    "vocal_sentiment": "Unavailable",
                    "valid": False
                }

            # Pacing calculation: chunk size is 3 seconds, so WPM = word_count * 20
            wpm = word_count * 20
            
            # Count filler words
            filler_words = ['um', 'uh', 'erm', 'err', 'like', 'you know', 'basically', 'actually', 'kind of']
            text_lower = transcript.lower()
            for filler in filler_words:
                pattern = r'\b' + re.escape(filler) + r'\b'
                filler_count += len(re.findall(pattern, text_lower))
            
            filler_detected = (filler_count > 0)
            
            # Vocal pitch variation heuristic
            pitch_score = 88
            vocal_sentiment = "Expressive"
            if wpm > 160:
                pitch_score -= 15
                vocal_sentiment = "Hurried"
            elif wpm < 90:
                pitch_score -= 10
                vocal_sentiment = "Monotone"
            
            if filler_count > 1:
                pitch_score -= min(15, filler_count * 5)
                vocal_sentiment = "Anxious"

            return {
                "wpm": wpm,
                "filler_word_detected": filler_detected,
                "transcript": transcript,
                "filler_count": filler_count,
                "pitch_score": max(40, min(100, pitch_score)),
                "vocal_sentiment": vocal_sentiment,
                "valid": True
            }

        return {
            "wpm": 0,
            "filler_word_detected": False,
            "transcript": "",
            "filler_count": 0,
            "pitch_score": None,
            "vocal_sentiment": "Unavailable",
            "valid": False
        }
        
    except Exception as e:
        print(f"[LIVE ERROR] Audio chunk analysis failed: {str(e)}")
        return {
            "wpm": 0,
            "filler_word_detected": False,
            "transcript": "",
            "filler_count": 0,
            "pitch_score": None,
            "vocal_sentiment": "Unavailable",
            "valid": False
        }


@phase_live_bp.route('/vision-status', methods=['GET'])
def get_vision_status():
    """
    Reports native CV dependency status for debugging live webcam analysis.
    """
    cascade_ready = False
    if OPENCV_AVAILABLE:
        init_cascades()
        cascade_ready = face_cascade is not None and eye_cascade is not None

    return jsonify({
        "opencv_available": OPENCV_AVAILABLE,
        "opencv_version": cv2.__version__ if OPENCV_AVAILABLE else None,
        "opencv_error": OPENCV_IMPORT_ERROR,
        "mediapipe_imported": MEDIAPIPE_IMPORTED,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "mediapipe_mode": MEDIAPIPE_MODE,
        "mediapipe_version": MEDIAPIPE_VERSION,
        "mediapipe_error": MEDIAPIPE_IMPORT_ERROR,
        "haar_cascades_ready": cascade_ready,
        "primary_analyzer": (
            f"mediapipe_{MEDIAPIPE_MODE}" if MEDIAPIPE_AVAILABLE else ("opencv_haar" if cascade_ready else None)
        )
    }), 200


# ===== WEBSOCKET SOCKET.IO EVENT HANDLERS =====

def init_socketio_events(socketio):
    """
    Binds WebSocket events to the Flask-SocketIO instance.
    """
    
    @socketio.on('connect', namespace='/ws/live-session')
    def on_connect():
        print(f"[CONN] Live presentation socket connected: {request.sid}")

    @socketio.on('disconnect', namespace='/ws/live-session')
    def on_disconnect():
        print(f"[CONN] Live presentation socket disconnected: {request.sid}")

    @socketio.on('start_session', namespace='/ws/live-session')
    def on_start_session(data):
        """
        Initializes a presentation practice session, pulls memory, and sends config.
        """
        user_id = data.get('user_id', 'guest')
        topic = data.get('topic', 'General Presentation').strip()
        
        print(f"[INFO] Creating live presentation session. Topic: {topic}, User: {user_id}")
        
        # Create DB session
        session = PresentationSession.create(user_id=user_id, topic=topic)
        
        # Context Memory Matrix: Fetch past reports for this topic
        historical_records = HistoricalReport.get_by_user_and_topic(user_id, topic)
        has_history = len(historical_records) > 0
        
        history_summary = {}
        if has_history:
            last_rep = historical_records[0].report_json
            history_summary = {
                "previous_score": last_rep.get("overall_score", 0),
                "top_strengths": last_rep.get("strengths", [])[:2],
                "top_recommendations": last_rep.get("recommendations", [])[:2]
            }

        socketio.emit('session_started', {
            "status": "success",
            "session_id": session.id,
            "has_history": has_history,
            "history_summary": history_summary
        }, room=request.sid, namespace='/ws/live-session')

    @socketio.on('video_frame', namespace='/ws/live-session')
    def on_video_frame(data):
        """
        Processes a real-time frame, updates metrics, and returns feedback.
        """
        session_id = data.get('session_id')
        frame_data = data.get('frame')
        
        if not session_id or not frame_data:
            return
            
        session = PresentationSession.get_by_id(session_id)
        if not session or session.status != 'STREAMING':
            return
            
        # Analyze frame
        metrics = analyze_webcam_frame(frame_data, session)

        if not metrics.get("valid", False):
            print(f"[LIVE VISION WARN] {metrics.get('hint', 'Face was not detected')}")
            socketio.emit('realtime_feedback', {
                "face_detected": False,
                "eye_contact": 0,
                "posture": 0,
                "confidence": 0,
                "hint": metrics.get("hint", "Face was not detected."),  # FIX: surface the reason to the UI
                "emotion": metrics.get("emotion", "NOT DETECTED")
            }, room=request.sid, namespace='/ws/live-session')
            return
        
        # Save only measured values to DB.
        session.update_metrics("eye_contact_scores", metrics["eye_contact"])
        session.update_metrics("posture_scores", metrics["posture"])
        session.update_metrics("confidence_scores", metrics["confidence"])
        
        # Send feedback in real-time
        socketio.emit('realtime_feedback', {
            "face_detected": True,
            "eye_contact": metrics["eye_contact"],
            "posture": metrics["posture"],
            "hint": metrics["hint"],
            "confidence": metrics["confidence"],
            "emotion": metrics["emotion"]
        }, room=request.sid, namespace='/ws/live-session')

    @socketio.on('audio_chunk', namespace='/ws/live-session')
    def on_audio_chunk(data):
        """
        Processes real-time audio chunk, checks filler density, triggers interruptions.
        """
        session_id = data.get('session_id')
        audio_data = data.get('audio')
        current_transcript = data.get('transcript_snippet', '').strip()
        
        if not session_id or not audio_data:
            return
            
        session = PresentationSession.get_by_id(session_id)
        if not session or session.status != 'STREAMING':
            return
            
        # Voice dynamics
        voice_metrics = analyze_audio_chunk(audio_data, session_id, current_transcript)
        if voice_metrics.get("valid", False):
            session.update_metrics("wpm_history", voice_metrics["wpm"])
            if voice_metrics.get("pitch_score") is not None:
                session.update_metrics("vocal_sentiment_scores", voice_metrics["pitch_score"])
        
        if voice_metrics.get("transcript"):
            session.update_metrics("transcripts", voice_metrics["transcript"])
            
        if voice_metrics.get("filler_count", 0) > 0:
            session.increment_metric("fillers_detected", voice_metrics["filler_count"])
            
        # Check Interruption Trigger
        # Condition: Trigger if filler count is high, or periodically every 45-60 seconds.
        # We also check if we are already in Q&A to avoid double triggers.
        total_fillers = session.metrics.get("fillers_detected", 0)
        interruptions = session.metrics.get("interruptions", [])
        
        # Frequency Limiter: At most 2 interruptions, and wait at least 45 seconds.
        can_interrupt = len(interruptions) < 2
        if can_interrupt and len(interruptions) > 0:
            last_int_time = datetime.fromisoformat(interruptions[-1]["timestamp"])
            if last_int_time.tzinfo is None:
                last_int_time = last_int_time.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_int_time).total_seconds()
            if elapsed < 45:
                can_interrupt = False
                
        # Trigger interruption on weak points (e.g. filler count reaches 5, or transcript shows poor pacing)
        trigger_by_filler = (total_fillers > 0 and total_fillers % 5 == 0)
        
        wpm_count = len(session.metrics.get("wpm_history", []))
        trigger_by_interval = wpm_count > 0 and wpm_count % 25 == 0
        if can_interrupt and (trigger_by_filler or trigger_by_interval):
            # Extract speaker's recent transcript context
            recent_transcripts = session.metrics.get("transcripts", [])[-4:]
            recent_speech = " ".join(recent_transcripts).strip() if recent_transcripts else session.topic

            # Generate high-level academic professor cross-question
            topic = session.topic
            question = f"Regarding {topic}, how do you validate your methodology against potential edge cases?"

            if gemini_available:
                try:
                    model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-3.6-flash'))
                    prompt = f"""You are Professor Eleanor Vance, a Senior Academic Evaluator and University Defense Chair presiding over a presentation.

PRESENTATION TOPIC: '{topic}'
SPEAKER'S RECENT WORDS: '{recent_speech}'

Formulate 1 sharp, highly educated, probing cross-examination question directly challenging or probing the speaker's claim, methodology, assumptions, or real-world applicability.
Your question MUST sound like a tough, inquisitive university professor testing their deep conceptual understanding. Keep it under 25 words."""
                    response = model.generate_content(prompt)
                    question = response.text.strip()
                except Exception as e:
                    print(f"[LIVE WARN] Gemini cross-question generation failed: {str(e)}")

            # Transition state in DB
            session.update_status("INTERRUPTED_Q&A")

            # Log interruption
            int_log = {
                "question": question,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "answer": None,
                "score": None
            }
            session.update_metrics("interruptions", int_log)

            # Emit interruption to frontend
            socketio.emit('interruption_trigger', {
                "question": question,
                "evaluator_name": "Prof. Eleanor Vance",
                "evaluator_role": "Senior Academic Defense Examiner"
            }, room=request.sid, namespace='/ws/live-session')

    @socketio.on('submit_answer', namespace='/ws/live-session')
    def on_submit_answer(data):
        """
        Receives user answer to panelist cross-question and grades it like a Senior Professor.
        """
        session_id = data.get('session_id')
        answer = data.get('answer', '').strip()

        if not session_id or not answer:
            return

        session = PresentationSession.get_by_id(session_id)
        if not session or session.status != 'INTERRUPTED_Q&A':
            return

        # Grade the answer like a senior academic examiner
        grade_score = 75
        feedback = "Answer recorded. Good effort."

        if gemini_available:
            try:
                model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-3.6-flash'))
                last_interruption = session.metrics["interruptions"][-1]
                question = last_interruption["question"]

                grading_prompt = f"""You are Professor Eleanor Vance, Senior Academic Examiner. Evaluate this student's response to your cross-examination question.

QUESTION: '{question}'
STUDENT ANSWER: '{answer}'

Evaluate the answer on:
1. Conceptual accuracy & analytical depth
2. Use of evidence/examples
3. Composure & defensive clarity

Return ONLY a single valid JSON object:
{{
    "score": <integer 0-100>,
    "feedback": "<2-sentence articulate critique explaining the score and how to make the answer bulletproof>"
}}"""
                response = model.generate_content(
                    grading_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                res_data = json.loads(response.text)
                grade_score = int(res_data.get("score", 75))
                feedback = res_data.get("feedback", "Articulate response.")
            except Exception as e:
                print(f"[LIVE WARN] Gemini grading failed: {str(e)}")
        
        # Update session logs — Firestore-compatible approach:
        # Read the current interruptions list, update the last unanswered entry
        # in-memory, then write the whole list back.
        try:
            ref = db.collection("presentation_sessions").document(session_id)
            doc = ref.get()
            if doc.exists:
                current_interruptions = doc.to_dict().get("metrics", {}).get("interruptions", [])
                # Find and update the last interruption that has no answer yet
                for i in range(len(current_interruptions) - 1, -1, -1):
                    if current_interruptions[i].get("answer") is None:
                        current_interruptions[i]["answer"] = answer
                        current_interruptions[i]["score"] = grade_score
                        current_interruptions[i]["feedback"] = feedback
                        break
                ref.update({"metrics.interruptions": current_interruptions})
                # Keep the in-memory session object in sync
                if session.metrics:
                    session.metrics["interruptions"] = current_interruptions
        except Exception as e:
            print(f"[LIVE WARN] Failed to update interruption answer in Firestore: {str(e)}")
        
        # Resume streaming status
        session.update_status("STREAMING")
        
        # Emit resolution
        socketio.emit('interruption_resolved', {
            "status": "success",
            "score": grade_score,
            "feedback": feedback
        }, room=request.sid, namespace='/ws/live-session')


# ===== COMPILATION REST ENDPOINT =====

@phase_live_bp.route('/submit', methods=['POST'])
@jwt_required(optional=True)
def submit_presentation():
    """
    Submits and finalizes the live presentation session:
    1. Compiles metrics from DB.
    2. Runs Gemini 1.5 Flash to structure a final 7Cs scorecard.
    3. Compares metrics with historical presentations on the same topic.
    4. Saves final report in MongoDB.
    """
    try:
        user_id = get_jwt_identity() or "guest"
        data = request.get_json()
        
        if not data or 'session_id' not in data:
            return jsonify({
                "error": "Missing session ID",
                "message": "Please provide a valid session_id"
            }), 400
            
        session_id = data['session_id']
        session = PresentationSession.get_by_id(session_id)
        
        if not session:
            return jsonify({
                "error": "Session not found",
                "message": "The requested session does not exist"
            }), 404
            
        # ===== STEP 1: COMPILE SESSION METRICS =====
        eye_scores = session.metrics.get("eye_contact_scores", [])
        posture_scores = session.metrics.get("posture_scores", [])
        wpm_history = session.metrics.get("wpm_history", [])
        fillers = session.metrics.get("fillers_detected", 0)
        interruptions = session.metrics.get("interruptions", [])
        transcripts = session.metrics.get("transcripts", [])
        
        confidence_scores = session.metrics.get("confidence_scores", [])
        vocal_sentiment_scores = session.metrics.get("vocal_sentiment_scores", [])

        def avg_or_zero(values):
            return int(sum(values) / len(values)) if values else 0

        avg_eye = avg_or_zero(eye_scores)
        avg_posture = avg_or_zero(posture_scores)
        avg_wpm = avg_or_zero(wpm_history)
        avg_confidence = avg_or_zero(confidence_scores)
        avg_vocal_pitch = avg_or_zero(vocal_sentiment_scores)

        qna_scores = [i["score"] for i in interruptions if i.get("score") is not None]
        avg_qna = avg_or_zero(qna_scores)

        # ── FIX: gate each dimension behind a minimum sample count, not just
        # "list is non-empty". One lucky frame or one hallucinated Whisper
        # chunk should not be enough evidence to score a whole dimension.
        has_enough_video = len(eye_scores) >= MIN_VALID_VIDEO_SAMPLES and len(posture_scores) >= MIN_VALID_VIDEO_SAMPLES
        has_enough_audio = len(wpm_history) >= MIN_VALID_AUDIO_SAMPLES

        visual_presence = int((avg_eye + avg_posture) / 2) if has_enough_video else None
        vocal_delivery = None
        if has_enough_audio:
            vocal_delivery = min(100, max(20, 100 - (fillers * 4) - abs(avg_wpm - 140) // 2))
        content_quality = avg_qna if qna_scores else None

        # ===== SCORING FORMULA WITH WEIGHT RENORMALIZATION =====
        # Base weights: Visual Presence (0.35), Vocal Delivery (0.35), Content / Q&A Quality (0.30).
        # If any component is unavailable (e.g. no Q&A interruptions occurred or camera was off),
        # its weight is redistributed proportionally among remaining active components:
        #   w_i_normalized = w_i / sum(w_active)
        # Full 3-component case: 0.35/1.0 = 35%, 0.35/1.0 = 35%, 0.30/1.0 = 30%
        # 2-component visual+vocal case: 0.35/0.70 = 50%, 0.35/0.70 = 50%
        weighted_scores = []
        if visual_presence is not None:
            weighted_scores.append((visual_presence, 0.35))
        if vocal_delivery is not None:
            weighted_scores.append((vocal_delivery, 0.35))
        if content_quality is not None:
            weighted_scores.append((content_quality, 0.30))

        total_weight = sum(weight for _, weight in weighted_scores)
        if total_weight > 0:
            # Proportional weight redistribution: sum( score * (w / total_weight) )
            normalized_scores = [(score, weight / total_weight) for score, weight in weighted_scores]
            overall_execution = int(round(sum(score * norm_w for score, norm_w in normalized_scores)))
        else:
            overall_execution = 0

        # ── FIX: explicit flag for "nothing measurable happened this session"
        # so the frontend can show "Insufficient data" instead of a bare 0/40
        # that looks like a real (bad) score.
        has_any_data = bool(weighted_scores)
        
        # Build full transcript for evaluation
        transcript_full = " ".join(transcripts)

        # ===== STEP 2: GENERATE 7Cs ANALYSIS VIA CENTRAL EVALUATOR =====
        from ai_evaluator import evaluate_7cs
        report_json = evaluate_7cs(
            text=transcript_full,
            module_type='live',
            context_metrics={
                "topic": session.topic,
                "avg_eye": avg_eye,
                "avg_posture": avg_posture,
                "avg_wpm": avg_wpm,
                "fillers": fillers,
                "avg_qna": avg_qna,
                "interruptions": interruptions,
                "overall_execution": overall_execution,
                "has_visual_metrics": has_enough_video,
                "has_voice_metrics": has_enough_audio,
                "has_qna_scores": bool(qna_scores)
            }
        )
        report_json["overall_score"] = overall_execution
        report_json["insufficient_data"] = not has_any_data  # FIX

        # Context Memory Matrix: Fetch past reports for this topic to check progress
        past_reports = HistoricalReport.get_by_user_and_topic(user_id, session.topic)
        
        comparison = {
            "improved": False,
            "difference": 0,
            "note": "First session on this topic recorded. Great start!"
        }
        
        if past_reports:
            last_overall = past_reports[0].report_json.get("overall_score", 0)
            diff = report_json["overall_score"] - last_overall
            comparison = {
                "improved": diff > 0,
                "difference": diff,
                "note": f"Your score shifted from {last_overall} to {report_json['overall_score']}. " + 
                        ("Keep polishing your delivery!" if diff <= 0 else "Great improvements in eye contact and presentation flow!")
            }

        # FIX: don't present a misleading "progress" comparison when this
        # session (or the prior one) had no real measured data.
        if not has_any_data:
            comparison = {
                "improved": False,
                "difference": 0,
                "note": "No usable camera or voice data was captured this session. Make sure your camera and mic are on, then try again."
            }

        report_json["comparison"] = comparison
        report_json["topic"] = session.topic
        report_json["session_metrics"] = {
            "avg_eye_contact": avg_eye,
            "avg_posture": avg_posture,
            "avg_wpm": avg_wpm,
            "total_fillers": fillers,
            "interruptions_handled": len(interruptions),
            "avg_confidence": avg_confidence,
            "avg_vocal_pitch": avg_vocal_pitch,
            "data_quality": {
                "video_samples": len(eye_scores),
                "audio_samples": len(wpm_history),
                "transcript_segments": len(transcripts),
                "qna_scores": len(qna_scores),
                "has_video_metrics": has_enough_video,
                "has_audio_metrics": has_enough_audio,
                "has_qna_scores": bool(qna_scores),
                "min_video_samples_required": MIN_VALID_VIDEO_SAMPLES,   # FIX
                "min_audio_samples_required": MIN_VALID_AUDIO_SAMPLES,   # FIX
            }
        }
        
        # Save historical report
        HistoricalReport.create(
            session_id=session_id,
            user_id=user_id,
            topic=session.topic,
            report_json=report_json
        )
        
        # Update session status
        session.update_status("FINISHED")
        
        return jsonify({
            "status": "success",
            "report": report_json
        }), 200
        
    except Exception as e:
        print(f"[LIVE ERROR] Final submit error: {str(e)}")
        return jsonify({
            "error": "Submission failed",
            "message": str(e)
        }), 500


@phase_live_bp.route('/history', methods=['GET'])
@jwt_required()
def get_topic_history():
    """
    Fetches historical live session reports for comparison dashboard.
    """
    try:
        user_id = get_jwt_identity()
        topic = request.args.get('topic', '').strip()
        
        if not topic:
            return jsonify({"error": "Missing topic"}), 400
            
        reports = HistoricalReport.get_by_user_and_topic(user_id, topic)
        return jsonify({
            "status": "success",
            "reports": [r.to_dict() for r in reports]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# touch reload
