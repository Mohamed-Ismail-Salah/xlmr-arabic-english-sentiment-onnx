import torch
import onnxruntime as ort
import numpy as np
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

 
model_path = "models/xlmr_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

 
text =".. . أعـظـم النعـم في الحياة راحة البال إن شعرت بها فأنت تملك كل شيء.."
inputs = tokenizer(text, return_tensors="pt")
inp = {k: v.cpu().numpy() for k, v in inputs.items()}

 
sess_fp32 = ort.InferenceSession("models/xlmr_sentiment.onnx")
sess_q = ort.InferenceSession("models/xlmr_sentiment_quantized.onnx")

 
def measure_onnx(sess, inp, runs=100):
    sess.run(None, inp)   
    start = time.time()
    for _ in range(runs):
        sess.run(None, inp)
    return (time.time() - start) / runs

def measure_pytorch(model, inputs, runs=100):
    model.eval()
    model.to("cpu")
    with torch.no_grad():
        _ = model(**inputs)   
        start = time.time()
        for _ in range(runs):
            _ = model(**inputs)
    return (time.time() - start) / runs

 
lat_pt = measure_pytorch(model, inputs)
lat_fp32 = measure_onnx(sess_fp32, inp)
lat_q = measure_onnx(sess_q, inp)

 
print(f" PyTorch latency: {lat_pt:.4f}s")
print(f" ONNX FP32 latency: {lat_fp32:.4f}s")
print(f" ONNX int8 latency: {lat_q:.4f}s")
print(f" Speed-up ONNX FP32 vs PyTorch: {lat_pt/lat_fp32:.2f}×")
print(f" Speed-up ONNX int8 vs PyTorch: {lat_pt/lat_q:.2f}×")
print(f" Speed-up quantized vs FP32 ONNX: {lat_fp32/lat_q:.2f}×")
