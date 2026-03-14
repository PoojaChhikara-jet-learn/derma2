"""
DermaWeb — Flask Analysis Server (Gemini Vision)
POST /analyze  { "image": "<base64 data-URL>", "save": true }
→ JSON skin analysis via Google Gemini Vision (FREE tier)

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY="AIza..."
    python app.py  →  http://localhost:5000

Get your FREE Gemini API key at:
    https://aistudio.google.com/app/apikey
"""

import os, io, base64, re, json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image

app = Flask(__name__, static_folder=".")
CORS(app)

SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Configure Gemini ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Use gemini-1.5-flash — free tier, supports vision
GEMINI_MODEL = "gemini-1.5-flash"

# ── Prompt ────────────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """You are DermaWeb AI, a professional dermatology image analysis assistant.
Analyse the skin image provided and return ONLY a valid JSON object — no markdown fences, no extra text, no explanation.

Return exactly this JSON structure:
{
  "condition": "<primary condition name, e.g. Melanoma, Rosacea, Eczema, Acne, Normal Skin, etc.>",
  "confidence": <integer 0-100>,
  "severity": "<Normal | Mild | Moderate | Severe>",
  "summary": "<2-sentence plain English summary of findings>",
  "observations": ["<clinical observation 1>", "<observation 2>", "<observation 3>"],
  "recommendations": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"],
  "urgency": "<Routine | Soon | Urgent>",
  "disclaimer": "This analysis is for informational and educational purposes only and does not constitute medical advice. Please consult a qualified dermatologist for diagnosis and treatment."
}

Rules:
- If the image is NOT a skin image, set condition to "Non-dermatological Image" and explain in summary.
- Be factual, clinical, and helpful.
- Return ONLY the JSON. No text before or after it."""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/script.js")
def js():
    return send_from_directory(".", "script.js")

@app.route("/health")
def health():
    return jsonify({
        "status"   : "ok",
        "api_ready": bool(GEMINI_API_KEY),
        "model"    : GEMINI_MODEL,
        "provider" : "Google Gemini",
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts: POST JSON { "image": "<base64 data-URL>", "save": true|false }
    Returns: JSON analysis result from Gemini Vision
    """

    if not GEMINI_API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY not set. "
                     "Get a free key at https://aistudio.google.com/app/apikey "
                     "then run: export GEMINI_API_KEY='AIza...'"
        }), 503

    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": 'Request body must contain { "image": "<base64>" }'}), 400

    # ── Decode base64 image ───────────────────────────────────────────────────
    try:
        b64 = payload["image"]
        mime = "image/jpeg"

        if "," in b64:
            header, b64 = b64.split(",", 1)
            m = re.search(r"data:([^;]+);", header)
            if m:
                mime = m.group(1)

        img_bytes = base64.b64decode(b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    # ── Save image (optional) ─────────────────────────────────────────────────
    saved = False
    if payload.get("save", False):
        try:
            ext   = mime.split("/")[-1].replace("jpeg", "jpg")
            fname = f"derma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
            pil_image.save(os.path.join(SAVE_DIR, fname))
            saved = True
            print(f"[SAVED] {fname}")
        except Exception as e:
            print(f"[WARN]  Could not save image: {e}")

    # ── Gemini Vision call ────────────────────────────────────────────────────
    try:
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            [ANALYSIS_PROMPT, pil_image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,        # low temperature = more consistent JSON
                max_output_tokens=1024,
            ),
        )

        raw = response.text.strip()

        # Strip any accidental markdown fences
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",           "", raw)
        raw = raw.strip()

        result = json.loads(raw)
        result["success"]  = True
        result["saved"]    = saved
        result["model"]    = GEMINI_MODEL
        result["provider"] = "Google Gemini"

        return jsonify(result)

    except json.JSONDecodeError as e:
        # Gemini returned non-JSON — return raw for debugging
        return jsonify({
            "error"     : f"Could not parse Gemini response as JSON: {e}",
            "raw_output": raw if 'raw' in dir() else "N/A",
        }), 500

    except Exception as e:
        err_msg = str(e)
        if "API_KEY" in err_msg or "credentials" in err_msg.lower():
            return jsonify({"error": "Invalid Gemini API key."}), 401
        if "quota" in err_msg.lower():
            return jsonify({"error": "Gemini API quota exceeded. Try again later."}), 429
        return jsonify({"error": f"Gemini analysis error: {err_msg}"}), 500


# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("━" * 52)
    print("  DermaWeb  |  Gemini Vision Analysis Server")
    print("━" * 52)
    if not GEMINI_API_KEY:
        print("  ⚠  GEMINI_API_KEY not set!")
        print("  →  Get free key: https://aistudio.google.com/app/apikey")
        print("  →  Then run:  export GEMINI_API_KEY='AIza...'")
    else:
        print(f"  ✓  Gemini API key found")
        print(f"  ✓  Model: {GEMINI_MODEL}")
    print("  →  http://localhost:5000")
    print("━" * 52)
    app.run(debug=True, host="0.0.0.0", port=5000)
