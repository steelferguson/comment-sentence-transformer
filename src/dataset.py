import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    """Custom dataset for multi-task learning."""
    def __init__(self, sentences, labels_classification, labels_sentiment, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels_classification = labels_classification
        self.labels_sentiment = labels_sentiment
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.sentences[idx], 
                                padding="max_length", 
                                truncation=True, 
                                max_length=self.max_length, 
                                return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "classification_label": torch.tensor(self.labels_classification[idx], dtype=torch.long),
            "sentiment_label": torch.tensor(self.labels_sentiment[idx], dtype=torch.long),
        }