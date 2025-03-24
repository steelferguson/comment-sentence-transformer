import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultiTaskDataset
from multi_task_distilbert import load_model
from sample_reviews import SAMPLE_DATA
from config import NUM_EPOCHS

# Load Model & Tokenizer
tokenizer, model = load_model()

# Define loss functions
classification_loss_fn = nn.CrossEntropyLoss()
sentiment_loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Example Data
sentences = SAMPLE_DATA['sentences']
labels_classification = SAMPLE_DATA['labels_classification']
labels_sentiment = SAMPLE_DATA['labels_sentiment']

# Create Dataset & DataLoader
dataset = MultiTaskDataset(sentences, labels_classification, labels_sentiment, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(NUM_EPOCHS):  
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        classification_logits = model(inputs, task="classification")
        sentiment_logits = model(inputs, task="sentiment")

        loss_classification = classification_loss_fn(classification_logits, batch["classification_label"].to(device))
        loss_sentiment = sentiment_loss_fn(sentiment_logits, batch["sentiment_label"].to(device))

        loss = loss_classification + loss_sentiment
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

# Save trained model (optional)
torch.save(model.state_dict(), "models/multi_task_model.pt")