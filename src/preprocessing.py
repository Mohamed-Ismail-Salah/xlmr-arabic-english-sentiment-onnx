import torch
 

# Custom Dataset class for sentiment analysis
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
         # Extract texts and labels from the DataFrame
        self.texts = dataframe["Text"].tolist()
        self.labels = dataframe["labels"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Return the total number of samples
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize the text with truncation and padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        # Return the input IDs, attention mask, and label as a dictionary
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
