from flask import Flask, request, jsonify
from gpt import GPTModelHandler
import torch
import os

app = Flask(__name__)

# Initialize model handler
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_model_handler = GPTModelHandler(device)

@app.route("/predict_gpt", methods=["POST"])
def predict_gpt():
    try:
        data = request.get_json()
        if "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
        prompt = data["prompt"]
        prediction = gpt_model_handler.predict(prompt)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)