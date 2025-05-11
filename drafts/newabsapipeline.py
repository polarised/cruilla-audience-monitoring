import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import spacy
import numpy as np
import pandas as pd
import random

# ------------------------------
# 1. Aspect Labels
# ------------------------------
ASPECTS = [
    "música", "sonido", "artistas", "comida", "bebidas", "precios",
    "colas", "organización", "seguridad", "ambiente", "instalaciones", "wifi"
]

# ------------------------------
# 2. Data Preparation
# ------------------------------
nlp = spacy.load("es_core_news_sm")
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

class FestivalDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.labels = ASPECTS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["tweet"]
        aspects = self.df.iloc[idx]["aspect"].split(",")
        label_vector = [1 if aspect.strip() in aspects else 0 for aspect in self.labels]
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        return inputs["input_ids"].squeeze(), inputs["attention_mask"].squeeze(), torch.tensor(label_vector, dtype=torch.float)

# ------------------------------
# 3. Model Definition
# ------------------------------
class AspectClassifier(nn.Module):
    def __init__(self, num_labels=len(ASPECTS)):
        super(AspectClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Use CLS token output
        return logits

# ------------------------------
# 4. Training Loop
# ------------------------------
def train_model(dataset_path, num_epochs=5, batch_size=16, lr=2e-5):
    # Load data
    df = pd.read_csv(dataset_path).dropna()
    train_dataset = FestivalDataset(df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model and optimizer
    model = AspectClassifier()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), "aspect_classifier.pt")
    print("✅ Model saved as 'aspect_classifier.pt'")

    # Generate sample predictions
    model.eval()
    samples = random.sample(list(df["tweet"][:100]), 20)
    results = []
    for tweet in samples:
        inputs = tokenizer(tweet, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.sigmoid(logits).squeeze().tolist()
            predicted_aspects = [ASPECTS[i] for i, p in enumerate(probs) if p > 0.5]
            results.append({"tweet": tweet, "predicted_aspects": ", ".join(predicted_aspects)})
    pd.DataFrame(results).to_csv("sample_predictions.csv", index=False)
    print("✅ Sample predictions saved as 'sample_predictions.csv'")

# ------------------------------
# 5. Run Training
# ------------------------------
if __name__ == "__main__":
    train_model("synthetic data/tweets_combinados.csv")
