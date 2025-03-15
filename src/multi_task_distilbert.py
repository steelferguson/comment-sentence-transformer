import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Load DistilBERT
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)

class MultiTaskDistilBERT(nn.Module):
    def __init__(self, transformer_model, num_classes_classification=4, num_classes_sentiment=3):
        super().__init__()
        self.encoder = transformer_model  # Shared backbone
        hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT

        # Task A: Sentence Classification (e.g., Topic Classification)
        self.classification_head = nn.Linear(hidden_size, num_classes_classification)

        # Task B: Sentiment Analysis (Positive, Negative, Neutral)
        self.sentiment_head = nn.Linear(hidden_size, num_classes_sentiment)

    def forward(self, inputs, task):
        # Get sentence embeddings (Mean Pooling)
        output = self.encoder(**inputs).last_hidden_state.mean(dim=1)

        if task == "classification":
            return self.classification_head(output)  # Logits for classification
        elif task == "sentiment":
            return self.sentiment_head(output)  # Logits for sentiment analysis