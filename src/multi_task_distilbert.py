import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskDistilBERT(nn.Module):
    def __init__(self, transformer_model, num_classes_classification=4, num_classes_sentiment=3):
        super().__init__()
        self.encoder = transformer_model  # Shared transformer backbone
        hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT

        # Task A: Sentence Classification (multi-class)
        self.classification_head = nn.Linear(hidden_size, num_classes_classification)

        # Task B: Sentiment Analysis (Positive, Negative, Neutral)
        self.sentiment_head = nn.Linear(hidden_size, num_classes_sentiment)

    def forward(self, inputs, task):
        """Forward pass, takes inputs and task type."""
        output = self.encoder(**inputs).last_hidden_state.mean(dim=1)  # Mean Pooling

        if task == "classification":
            return self.classification_head(output)  # Multi-class logits
        elif task == "sentiment":
            return self.sentiment_head(output)  # Sentiment logits

def load_model(model_name="distilbert-base-uncased"):
    """Helper function to load tokenizer and model."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = AutoModel.from_pretrained(model_name)
    model = MultiTaskDistilBERT(transformer)
    return tokenizer, model