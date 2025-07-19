import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_path = "models/xlmr_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


inputs = tokenizer(".. . أعـظـم النعـم في الحياة راحة البال إن شعرت بها فأنت تملك كل شيء..", return_tensors="pt")
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "xlmr_sentiment.onnx",
    input_names=["input_ids","attention_mask"],
    output_names=["logits"],
   dynamic_axes={
    "input_ids": {0: "batch", 1: "sequence"},
    "attention_mask": {0: "batch", 1: "sequence"},
    "logits": {0: "batch"}
},
    opset_version=17,
    do_constant_folding=True,
)
print("created xlmr_sentiment.onnx")
