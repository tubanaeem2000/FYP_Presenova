import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

def p(*args):
    print(*args, flush=True)

p("=== 1. VERIFYING CASCADES IN PHASE_LIVE ===")
from phase_live import init_cascades, face_cascade, eye_cascade, analyze_webcam_frame
import phase_live

init_cascades()
p("face_cascade loaded:", phase_live.face_cascade is not None and not phase_live.face_cascade.empty())
p("eye_cascade loaded:", phase_live.eye_cascade is not None and not phase_live.eye_cascade.empty())

p("=== 2. VERIFYING FRAME ANALYSIS ===")
# Create a test synthetic image (320x240)
test_img = np.zeros((240, 320, 3), dtype=np.uint8)
# Draw a simulated face (white oval)
cv2.ellipse(test_img, (160, 100), (45, 60), 0, 0, 360, (200, 200, 200), -1)
# Draw simulated eyes
cv2.circle(test_img, (145, 90), 8, (50, 50, 50), -1)
cv2.circle(test_img, (175, 90), 8, (50, 50, 50), -1)

_, buffer = cv2.imencode('.jpg', test_img)
import base64
frame_b64 = base64.b64encode(buffer).decode('utf-8')

res = analyze_webcam_frame(frame_b64)
p("Frame analysis result keys:", list(res.keys()))
p("Face detected:", res.get("face_detected"))
p("Eye contact score:", res.get("eye_contact"))
p("Posture score:", res.get("posture"))
p("Hint:", res.get("hint"))

p("=== 3. VERIFYING AI EVALUATOR (LIVE MODULE) ===")
from ai_evaluator import evaluate_7cs

context_metrics = {
    "topic": "AI in Healthcare",
    "avg_eye": 75,
    "avg_posture": 82,
    "avg_wpm": 135,
    "fillers": 2,
    "avg_qna": 85,
    "interruptions": [{"question": "What is the error rate?", "score": 85}],
    "overall_execution": 81,
    "has_visual_metrics": True,
    "has_voice_metrics": True,
    "has_qna_scores": True
}

transcript = "Good morning everyone. Today I will present our research on AI in healthcare. We tested deep learning models with significant precision."
report = evaluate_7cs(transcript, module_type="live", context_metrics=context_metrics)

p("Evaluator Report Overall Score:", report.get("overall_score"))
p("Category Scores:", report.get("category_scores"))
p("7Cs Scores:", report.get("seven_cs_scores"))
p("Strengths count:", len(report.get("strengths", [])))
p("Recommendations count:", len(report.get("recommendations", [])))

assert report.get("overall_score") is not None, "Overall score is None!"
assert report.get("category_scores") is not None, "Category scores are None!"
assert "Structure" in report.get("category_scores", {}), "Structure missing in category scores!"
assert report.get("seven_cs_scores") is not None, "seven_cs_scores is None!"
assert "Clear" in report.get("seven_cs_scores", {}), "Clear missing in seven_cs_scores!"

p("=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
