# XLMR Arabic-English Sentiment ONNX

This project presents a **production-ready multilingual sentiment analysis system**, capable of understanding **Arabic and English** text, and optimized for **fast and lightweight deployment** using **ONNX** and **INT8 quantization**.

 

---

## 🎯 Project Objectives

- Build a **sentiment analysis model** that supports **Arabic and English**.
- Fine-tune a multilingual transformer model on custom sentiment data.
- Convert the trained model from **PyTorch to ONNX** format.
- Apply **model quantization** to optimize for speed and size.
- Benchmark performance between PyTorch, ONNX FP32, and ONNX INT8.
- Prepare the model for integration into web/mobile applications.

---

## 🛠️ Technologies Used

| Tool            | Purpose                           |
|-----------------|------------------------------------|
| HuggingFace Transformers | Pretrained XLM-RoBERTa model |
| PyTorch         | Model fine-tuning                  |
| ONNX            | Model export                       |
| ONNX Runtime    | Fast inference engine              |
| Quantization    | Model size/speed optimization      |
| Tokenizers      | Multilingual text preprocessing    |
| Python          | Implementation                     |

---

## 📊 Model Overview

- **Base model**: `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`
- **Languages supported**: Arabic, English
- **Sentiment classes**: Positive, Neutral, Negative
- **Model format**: Trained in PyTorch → Exported to ONNX → Quantized to INT8
- **Deployment ready**: Fast inference via ONNXRuntime

---

## 🚀 Implementation Stages

### 1. Fine-tuning the model
- Trained on mixed Arabic-English labeled sentiment dataset.
- Used `AutoModelForSequenceClassification` with softmax output.
- Applied appropriate tokenization and padding for both languages.

### 2. Export to ONNX
- Converted model using `torch.onnx.export` with dynamic axes.
- Ensured compatibility with batch inputs and inference APIs.

### 3. Quantization
- Applied post-training dynamic quantization using `onnxruntime.quantization`.
- Reduced model size and improved inference time (up to **2x speed-up**).

### 4. Benchmarking
- Created Python script to compare:
  - PyTorch inference
  - ONNX FP32
  - ONNX INT8
- Measured average latency over 100 runs.

---

## ⚡ Performance Results

| Format         | Latency (s) | Speed-up vs PyTorch |
|----------------|-------------|----------------------|
| PyTorch        | 0.0947      | 1×                   |
| ONNX FP32      | 0.0636      | 1.49×                |
| ONNX Quantized | 0.0458      | 2.07×                |

> ✅ Quantized model is 2× faster and significantly smaller.

---

## 📂 Project Structure

```bash
Sentiment Analysis/
├── models/                # Original + ONNX + quantized models
├── deployment/            # Scripts for export, quantization, benchmarking
├── data/                  # (Optional) Training/Evaluation data
├── export_onnx.py
├── quantize.py
├── benchmark.py
├── requirements.txt
└── README.md
