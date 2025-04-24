from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt import GPTModelHandler
import torch

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_model_handler = GPTModelHandler(device)

@app.route("/predict_gpt", methods=["POST"])
def predict_gpt():
    try:
        data = request.get_json(force=True)
        print("Received request:", data)
        if "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
        prompt = data["prompt"]
        prediction = gpt_model_handler.predict(prompt)
        print("Returning prediction:", prediction)
        return jsonify(prediction)
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
