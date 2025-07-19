from transformers import AutoModelForSequenceClassification

def load_model(model_name: str = "xlm-roberta-base", num_labels: int = 3):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
