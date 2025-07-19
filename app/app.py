from flask import Flask, request, jsonify
import onnxruntime as ort
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import helpers

 
app = Flask(__name__)

 # Load the tokenizer, model, and ONNX session
tokenizer = AutoTokenizer.from_pretrained("./models/xlmr_model")   
onnx_path = "./models/xlmr_sentiment_quantized.onnx"
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


 
@app.route("/predict", methods=["POST"])
 # Handles a POST request with input text and returns predicted sentiment as JSON
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON."}), 400

    text = data["text"]
    label = helpers.predict_sentiment(text,tokenizer=tokenizer, session=session)
    return jsonify({"text": text, "sentiment": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
