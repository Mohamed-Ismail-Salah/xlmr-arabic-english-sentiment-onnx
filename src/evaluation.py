import torch
from sklearn.metrics import classification_report

# Evaluation function to assess the model's performance on a validation/test dataset
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    print("Evaluation Results:")
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"]))
