import os, io, base64, re, json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image

# Flask setup
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Gemini Configuration ─────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-1.5-flash"

# ── Prompt ───────────────────────────────────────
ANALYSIS_PROMPT = """
You are DermaWeb AI, a professional dermatology image analysis assistant.

Return ONLY valid JSON:

{
  "condition": "<condition>",
  "confidence": <0-100>,
  "severity": "<Normal | Mild | Moderate | Severe>",
  "summary": "<2 sentence explanation>",
  "observations": ["obs1","obs2","obs3"],
  "recommendations": ["rec1","rec2","rec3"],
  "urgency": "<Routine | Soon | Urgent>",
  "disclaimer": "This analysis is informational only."
}

Rules:
- If image is NOT skin → condition="Non-dermatological Image"
- Return ONLY JSON
"""

# ── Routes ───────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "api_ready": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
        "provider": "Google Gemini"
    })


@app.route("/analyze", methods=["POST"])
def analyze():

    if not GEMINI_API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY not set"
        }), 503

    payload = request.get_json()

    if not payload or "image" not in payload:
        return jsonify({
            "error": "Request must contain base64 image"
        }), 400

    # ── Decode image ─────────────────────────────
    try:
        b64 = payload["image"]

        if "," in b64:
            _, b64 = b64.split(",", 1)

        img_bytes = base64.b64decode(b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    # ── Save image if requested ──────────────────
    saved = False
    if payload.get("save", False):
        try:
            fname = f"derma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            pil_image.save(os.path.join(SAVE_DIR, fname))
            saved = True
        except Exception as e:
            print("Save error:", e)

    # ── Gemini Vision Analysis ───────────────────
    try:

        model = genai.GenerativeModel(GEMINI_MODEL)

        response = model.generate_content(
            [ANALYSIS_PROMPT, pil_image],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1024
            }
        )

        raw = response.text.strip()

        raw = re.sub(r"^```[a-zA-Z]*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

        result = json.loads(raw)

        result["success"] = True
        result["saved"] = saved
        result["model"] = GEMINI_MODEL
        result["provider"] = "Google Gemini"

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({
            "error": "Gemini returned invalid JSON",
            "raw": raw
        }), 500

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ── Startup ──────────────────────────────────────
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("DermaWeb Server Starting...")
    print("Model:", GEMINI_MODEL)
    print("Port:", port)

    app.run(host="0.0.0.0", port=port)
