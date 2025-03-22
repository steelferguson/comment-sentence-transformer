import torch
from multi_task_distilbert import load_model
from sample_reviews import SAMPLE_SET

def predict(sentence):
    tokenizer, model = load_model()
    ## use our own traine models once we finalize that piece
    # tokenizer, model = load_model(model_path="models/multi_task_model.pt")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        classification_logits = model(inputs, task="classification")
        sentiment_logits = model(inputs, task="sentiment")

    classification_pred = torch.argmax(classification_logits, dim=1).item()
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()

    return classification_pred, sentiment_pred

sentence = "I am undecided about my sentiment!"

for sentence in SAMPLE_SET:
    print(sentence)
    print(predict(sentence))