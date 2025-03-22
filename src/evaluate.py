import torch
from multi_task_distilbert import load_model
from sample_reviews import SAMPLE_SET

# Load model and tokenizer
tokenizer, model = load_model()
model.eval()


# Test sentence
for test_sentence in SAMPLE_SET:
    inputs = tokenizer(test_sentence, padding=True, truncation=True, return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        classification_logits = model(inputs, task="classification")
        sentiment_logits = model(inputs, task="sentiment")

    # Convert to predictions
    classification_pred = torch.argmax(classification_logits, dim=1).item()
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()

    print(f"test_sentence: {test_sentence}")
    print(f"Predicted Class: {classification_pred}")
    print(f"Predicted Sentiment: {sentiment_pred}")

