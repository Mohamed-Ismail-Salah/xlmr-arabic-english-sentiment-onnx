# XLM-R Sentiment Analysis (Arabic-English, ONNX Optimized)

This project presents a **multilingual sentiment analysis pipeline**, capable of understanding **Arabic and English** text, and optimized for **fast and lightweight deployment** using **ONNX** and **INT8 quantization**.  
The system is production-ready and benchmarks different inference formats for performance comparison.

---

## 🎯 Project Objectives

- Build a robust **sentiment analysis model** for Arabic and English.
- Compare multiple pretrained models:  
  - `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`  
  - `xlm-roberta-base`  
  → **Selected `xlm-roberta-base`** based on accuracy on custom dataset.
- Fine-tune the selected model on multilingual sentiment data.
- Convert the model from **PyTorch → ONNX**.
- Apply **quantization** for faster and smaller inference models.
- Benchmark inference speed across PyTorch, ONNX FP32, and ONNX INT8.
- Use separate **sources for training and evaluation data** to ensure generalization.

---

## 🛠️ Technologies Used

| Tool                     | Purpose                           |
|--------------------------|------------------------------------|
| HuggingFace Transformers| Fine-tuning pretrained XLM-R models|
| PyTorch                  | Model training                     |
| ONNX                     | Model export format                |
| ONNX Runtime             | Efficient inference                |
| Quantization             | Speed and size optimization        |
| Tokenizers               | Multilingual preprocessing         |
| Python                   | Implementation & scripting         |

---

## 📊 Model Overview

- **Base model used**: `xlm-roberta-base`
- **Compared against**: `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`
- **Selected based on**: Accuracy on custom Arabic-English sentiment dataset
- **Sentiment classes**: Positive, Neutral, Negative
- **Languages supported**: Arabic, English
- **Model pipeline**:
  - Trained with PyTorch
  - Exported to ONNX
  - Quantized to INT8
- **Optimized for deployment**: Fast inference with ONNXRuntime

---

## 🚀 Implementation Stages

### 1. Model Comparison & Fine-tuning
- Evaluated multiple pretrained multilingual models.
- Selected `xlm-roberta-base` based on performance.
- Fine-tuned using a custom dataset containing Arabic and English sentiment labels.
- Data collected from **varied sources**, and evaluation was performed using a **different source** to assess generalization.

### 2. ONNX Export
- Exported PyTorch model using `torch.onnx.export`.
- Used `dynamic_axes` to support variable batch sizes.
- Defined clear input/output names for easier deployment.

### 3. Quantization
- Applied **post-training dynamic quantization** using `onnxruntime.quantization`.
- Significantly reduced model size and improved inference time (~2× speed-up over PyTorch).

### 4. Benchmarking
- Created benchmarking script to compare:
  - PyTorch inference
  - ONNX FP32
  - ONNX INT8 (quantized)
- Measured average latency over 100 runs for fair comparison.

---

## ⚡ Performance Results

| Format         | Latency (s) | Speed-up vs PyTorch |
|----------------|-------------|----------------------|
| PyTorch        | 0.0947      | 1×                   |
| ONNX FP32      | 0.0636      | 1.49×                |
| ONNX INT8      | 0.0458      | 2.07×                |

> ✅ The quantized model is nearly **2× faster** and significantly smaller in size, ideal for real-time applications.

---

## 📂 Project Structure

```bash
Sentiment Analysis/
├── deployment/            # Scripts: export, quantize, benchmark, inference
├── models/                # (Ignored)Trained PyTorch model + ONNX + INT8 versions
├── notebooks/             # Training notebooks and exploration
├── data/                  # (Ignored) Raw and preprocessed datasets
├── venv/                  # (Ignored) Python virtual environment
├── .gitignore
├── requirements.txt
└── README.md
