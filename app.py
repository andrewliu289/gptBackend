from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt import GPTModelHandler
import torch

import base64, io, os, json, random, importlib.util
from pathlib import Path
from PIL import Image, ImageOps
import cv2
from nb_classifier import classify
from image_utils import (
    pil_to_base64, base64_to_pil,
    FILTERS, list_presets, load_preset
)
from helpers_nltk import synonym, random_quote

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_model_handler = GPTModelHandler(device)

@app.route("/predict_gpt", methods=["POST"])
def predict_gpt():
    try:
        print("NEW REQUEST")
        print("Headers:", dict(request.headers))
        print("Raw data:", request.data)
        
        data = request.get_json(force=True)
        print("Parsed JSON:", data)

        if "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400

        prompt = data["prompt"]
        prediction = gpt_model_handler.predict(prompt)
        print("Prediction:", prediction)

        return jsonify(prediction)
    except Exception as e:
        print("EXCEPTION:", str(e))
        return jsonify({"error": str(e)}), 500
    
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200
# ----------------------------------------------------------------------------
@app.route("/smart_reply", methods=["POST"])
def smart_reply():
    """
    • Classify prompt as 'story', 'joke', or 'fact' via Naïve Bayes
    • Rewrite prompt for GPT accordingly
    • Return GPT reply + intent + a fun NLTK synonym / quote hook
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    intent = classify(prompt)  # story | joke | fact
    system_prompt = {
        "story": f"Write a short imaginative story about: {prompt}",
        "joke" : f"Tell one witty, family‑friendly joke about: {prompt}",
        "fact" : prompt,
    }[intent]

    # use the SAME GPT handler you already instantiated below
    gpt_reply = gpt_model_handler.predict(system_prompt)["prediction_gpt"]

    # sprinkle a tiny NLTK surprise
    sparkle = {
        "story": f"(Here’s a cool synonym for “wonder” → {synonym('wonder')})",
        "joke" : f"(Random quote: “{random_quote()}”)",
        "fact" : "",
    }[intent]

    return jsonify({"reply": gpt_reply, "intent": intent, "extra": sparkle})

@app.post("/upload_image")
def upload_image():
    """Echo the user’s image back as base64 so the frontend can render it."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    img = Image.open(f.stream)
    return jsonify({"image_b64": pil_to_base64(img)})

@app.post("/filter_image")
def filter_image():
    """Apply a simple filter (gray | edge | neon) defined in image_utils.py"""
    data = request.get_json(force=True)
    img_b64, flt = data.get("image_b64"), data.get("filter")
    if not img_b64 or flt not in FILTERS:
        return jsonify({"error": "bad request"}), 400
    out = FILTERS[flt](base64_to_pil(img_b64))
    return jsonify({"image_b64": pil_to_base64(out)})

@app.get("/preset/<name>")
def preset_image(name):
    """Serve one of the stock demo images stored in backend/data/presets/"""
    try:
        return jsonify({"image_b64": pil_to_base64(load_preset(name))})
    except FileNotFoundError:
        return jsonify({"error": "preset not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
