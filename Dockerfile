
FROM python:3.10-slim
WORKDIR /app

COPY app/ /app/app/
COPY models/xlmr_model /app/models/xlmr_model
COPY models/xlmr_sentiment_quantized.onnx /app/models/xlmr_sentiment_quantized.onnx
COPY requirements.txt /app/
RUN pip install torch==2.4.1
RUN pip install --no-cache-dir -r requirements.txt



CMD ["python", "app/app.py"]



 
 
     
 