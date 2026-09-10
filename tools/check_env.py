import os
import sys

def p(*args):
    print(*args, flush=True)

import cv2
p("CV2 Version:", cv2.__version__)
cascade_dir = getattr(cv2.data, "haarcascades", "")
p("Cascade dir:", cascade_dir)
face_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(cascade_dir, "haarcascade_eye.xml")
p("Face cascade exists:", os.path.exists(face_path))
p("Eye cascade exists:", os.path.exists(eye_path))

if os.path.exists(face_path) and os.path.exists(eye_path):
    face_c = cv2.CascadeClassifier(face_path)
    eye_c = cv2.CascadeClassifier(eye_path)
    p("Face cascade empty?:", face_c.empty())
    p("Eye cascade empty?:", eye_c.empty())

# Test Gemini
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

for m in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-3.6-flash"]:
    p(f"--- Testing model: {m} ---")
    try:
        model = genai.GenerativeModel(m)
        res = model.generate_content("Say hello in one word", request_options={"timeout": 5})
        p(f"Result for {m}:", res.text.strip())
    except Exception as e:
        p(f"Error for {m}:", e)

# Test Groq
from groq import Groq
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=5.0)
    models = groq_client.models.list()
    p("Groq models OK, found:", len(models.data))
except Exception as e:
    p("Groq error:", e)
