import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_scheduler
from tqdm import tqdm
import pandas as pd

from model import load_model
from dataset import SentimentDataset
from evaluate import evaluate

def train():
     # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       # Model and training hyperparameters
    model_name = "xlm-roberta-base"
    batch_size = 16
    lr = 2e-5
    epochs = 3

    # Load training and validation datasets
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/test.csv")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(model_name).to(device)

    # Create dataset and dataloaders
    train_ds = SentimentDataset(train_df, tokenizer)
    val_ds = SentimentDataset(val_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Set up optimizer and learning rate scheduler 
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_loader))

    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, val_loader, device)

  # Save the trained model and tokenizer
    model.save_pretrained("models/xlmr_model")
    tokenizer.save_pretrained("models/xlmr_model")

if __name__ == "__main__":
    train()
