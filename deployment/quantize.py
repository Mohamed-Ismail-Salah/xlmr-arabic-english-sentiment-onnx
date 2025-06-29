from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="models/xlmr_sentiment.onnx",
    model_output="models/xlmr_sentiment_quantized.onnx",
    weight_type=QuantType.QInt8
)