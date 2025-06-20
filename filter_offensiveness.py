import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import re
import random
import numpy as np
import wandb

# === 0. Setup and Seeding ===
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# === 1. Initialize WandB ===
wandb.init(
    project="classifier_1_cruilla",
    config={
        "model": "bert-base-uncased",
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 5e-5,
        "max_length": 128,
    }
)
config = wandb.config

# === 2. Load and Preprocess Data ===
def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

# Load and merge datasets
festival_data = pd.read_csv("combned_output.csv").dropna(subset=["fullText", "filter1_label"])
offensive_data = pd.read_csv("burradas_catcas.csv")  # Should contain 'text' and 'is_offensive' columns

# Merge datasets
data = pd.merge(
    festival_data,
    offensive_data[["text", "is_offensive"]],
    left_on="fullText",
    right_on="text",
    how="left"
)
data["is_offensive"] = data["is_offensive"].fillna(0)  # Assume non-offensive if unknown
data["clean_text"] = data["fullText"].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
data["label_encoded"] = label_encoder.fit_transform(data["filter1_label"])

# Train-test split
X_train, X_val, y_train_festival, y_val_festival, y_train_offensive, y_val_offensive = train_test_split(
    data["clean_text"],
    data["label_encoded"],
    data["is_offensive"],
    test_size=0.2,
    random_state=42,
    stratify=data[["label_encoded", "is_offensive"]]
)

# === 3. Tokenization and Dataset ===
tokenizer = BertTokenizer.from_pretrained(config.model)

def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=config.max_length,
        return_tensors="pt"
    )

class MultiTaskDataset(Dataset):
    def __init__(self, encodings, festival_labels, offensive_labels):
        self.encodings = encodings
        self.festival_labels = torch.tensor(festival_labels.values)
        self.offensive_labels = torch.tensor(offensive_labels.values).float()
        
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["festival_labels"] = self.festival_labels[idx]
        item["offensive_labels"] = self.offensive_labels[idx]
        return item
    
    def __len__(self):
        return len(self.festival_labels)

train_encodings = tokenize_texts(X_train)
val_encodings = tokenize_texts(X_val)

train_dataset = MultiTaskDataset(train_encodings, y_train_festival, y_train_offensive)
val_dataset = MultiTaskDataset(val_encodings, y_val_festival, y_val_offensive)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

# === 4. Model Architecture ===
class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config, num_festival_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.offensive_classifier = nn.Linear(config.hidden_size, 1)
        self.festival_classifier = nn.Linear(config.hidden_size, num_festival_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        offensive_logits = self.offensive_classifier(pooled_output)
        festival_logits = self.festival_classifier(pooled_output)
        
        return {
            "offensive_logits": offensive_logits,
            "festival_logits": festival_logits
        }

model = MultiTaskBERT.from_pretrained(
    config.model,
    num_festival_labels=len(label_encoder.classes_)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 5. Loss Function and Optimizer ===
def compute_loss(outputs, batch):
    offensive_loss = nn.BCEWithLogitsLoss()(
        outputs["offensive_logits"].squeeze(),
        batch["offensive_labels"]
    )
    festival_loss = nn.CrossEntropyLoss()(
        outputs["festival_logits"],
        batch["festival_labels"]
    )
    return offensive_loss + festival_loss

optimizer = AdamW(model.parameters(), lr=config.learning_rate)
num_training_steps = len(train_loader) * config.epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# === 6. Training Loop ===
for epoch in range(config.epochs):
    model.train()
    total_loss = 0
    train_festival_preds, train_festival_labels = [], []
    train_offensive_preds, train_offensive_labels = [], []
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        loss = compute_loss(outputs, batch)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # Festival predictions
        festival_preds = torch.argmax(outputs["festival_logits"], dim=1)
        train_festival_preds.extend(festival_preds.cpu().numpy())
        train_festival_labels.extend(batch["festival_labels"].cpu().numpy())
        
        # Offensive predictions
        offensive_probs = torch.sigmoid(outputs["offensive_logits"].squeeze())
        train_offensive_preds.extend((offensive_probs > 0.5).int().cpu().numpy())
        train_offensive_labels.extend(batch["offensive_labels"].int().cpu().numpy())
        
        loop.set_postfix(loss=loss.item())
    
    # === Validation Phase ===
    model.eval()
    val_loss = 0
    val_festival_preds, val_festival_labels = [], []
    val_offensive_preds, val_offensive_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            val_loss += compute_loss(outputs, batch).item()
            
            # Festival predictions
            festival_preds = torch.argmax(outputs["festival_logits"], dim=1)
            val_festival_preds.extend(festival_preds.cpu().numpy())
            val_festival_labels.extend(batch["festival_labels"].cpu().numpy())
            
            # Offensive predictions
            offensive_probs = torch.sigmoid(outputs["offensive_logits"].squeeze())
            val_offensive_preds.extend((offensive_probs > 0.5).int().cpu().numpy())
            val_offensive_labels.extend(batch["offensive_labels"].int().cpu().numpy())
    
    # === Metrics Calculation ===
    # Festival metrics
    festival_train_acc = accuracy_score(train_festival_labels, train_festival_preds)
    festival_val_acc = accuracy_score(val_festival_labels, val_festival_preds)
    
    # Offensive metrics
    offensive_train_acc = accuracy_score(train_offensive_labels, train_offensive_preds)
    offensive_val_acc = accuracy_score(val_offensive_labels, val_offensive_preds)
    
    # === WandB Logging ===
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": total_loss / len(train_loader),
        "val_loss": val_loss / len(val_loader),
        "festival_train_acc": festival_train_acc,
        "festival_val_acc": festival_val_acc,
        "offensive_train_acc": offensive_train_acc,
        "offensive_val_acc": offensive_val_acc,
    })
    
    print(f"\nEpoch {epoch + 1}")
    print(f"Train Loss: {total_loss / len(train_loader):.4f}")
    print(f"Festival Train Acc: {festival_train_acc:.4f} | Val Acc: {festival_val_acc:.4f}")
    print(f"Offensive Train Acc: {offensive_train_acc:.4f} | Val Acc: {offensive_val_acc:.4f}")

# === 7. Save Model ===
model.save_pretrained("./classifier_1")
tokenizer.save_pretrained("./classifier_1")
wandb.finish()

# === 8. Inference Example ===
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    offensive_prob = torch.sigmoid(outputs["offensive_logits"].squeeze()).item()
    festival_pred = torch.argmax(outputs["festival_logits"]).item()
    
    return {
        "text": text,
        "is_offensive": offensive_prob > 0.5,
        "offensive_score": offensive_prob,
        "festival_class": label_encoder.inverse_transform([festival_pred])[0]
    }

# Example usage
print(predict("This festival was amazing!"))
print(predict("I hate these people!"))
