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
| Docker                   | Containerized deployment           |
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
- **Docker support**: The full API is containerized using **Docker** for easy and fast deployment with one command.

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

### 5. Docker Deployment
- Containerized the entire API using **Docker** to simplify deployment.
- Dockerfile installs dependencies, loads the quantized ONNX model, and launches a lightweight Flask server.
- To run the API locally:
  ```bash
  docker build -t sentiment-api .
  docker run --rm -p 5000:5000 sentiment-api
---
## ⚡ Performance Results

| Format         | Latency (s) | Speed-up vs PyTorch |
|----------------|-------------|----------------------|
| PyTorch        | 0.0947      | 1×                   |
| ONNX FP32      | 0.0636      | 1.49×                |
| ONNX INT8      | 0.0458      | 2.07×                |

> ✅ The quantized model is nearly **2× faster** and significantly smaller in size, ideal for real-time applications.

---

## 🧪 Benchmark Comparison of Sentiment Models

| **Model**                                   | **Accuracy** | **Macro F1 Score** | **Negative F1** | **Neutral F1** | **Positive F1** | **Support** |
|--------------------------------------------|--------------|---------------------|------------------|----------------|------------------|-------------|
| **XLM-RoBERTa Base**                        | 0.73         | 0.73                | 0.72             | 0.74           | 0.72             | 15,090      |
| **Twitter XLM-R Sentiment (Multilingual)** | 0.70         | 0.70                | 0.74             | 0.62           | 0.74             | 7,800       |

## 🧱 Docker Deployment

You can easily run the sentiment analysis API using Docker without installing any dependencies.

### 🔧 How to Run the Project with Docker

1. **Clone the repository (if not already):**

 2. **Build the Docker image:**
    run--> docker build -t sentiment-api .
   
3. **Run the API container:**
   run-->docker run --rm -p 5000:5000 sentiment-api


## 📂 Project Structure

```bash
Sentiment Analysis/
├── optimization/            # Scripts: export, quantize, benchmark, inference
├── models/                # (Ignored)Trained PyTorch model + ONNX + INT8 versions
├── notebooks/             # Training notebooks and exploration
├── data/                  # (Ignored) Raw and preprocessed datasets
├── venv/
├── app/
├── src/             
├── .gitignore
├── requirements.txt
├──  Dockerfile
└── README.md
