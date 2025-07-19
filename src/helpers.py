import numpy as np
 
 
# Predicts the sentiment of the input text and returns the corresponding label (Negative, Neutral, Positive)
def predict_sentiment(text,tokenizer, session):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
    ort_outs = session.run(None, ort_inputs)
    logits = ort_outs[0]
    pred = np.argmax(logits, axis=1)[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[pred]