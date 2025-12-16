"""
Custom Text Classification Model Training System.
Trains a DistilBERT model on user-provided dataset with custom labels.
No suicide detection — only labels from the dataset are used.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from torch.utils.data import Dataset
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class TextDataset(Dataset):
    """PyTorch dataset for text classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class CustomTextModelTrainer:
    """Train and save a custom text classification model."""

    def __init__(self, model_dir: str = "./custom_text_model"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.unique_labels = []

    def load_and_clean_dataset(
        self, csv_path: str, text_column: str = "Text", label_column: str = "Mental_Health_Status"
    ) -> Tuple[List[str], List[str]]:
        """Load CSV, clean text, AND BALANCE the dataset."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed")

        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Columns not found. Available: {df.columns}")

        # Remove rows with missing values
        df = df.dropna(subset=[text_column, label_column])

        # 1. Clean Text (Lowercase + Strip) - MATCHING PREDICTION LOGIC
        df[text_column] = df[text_column].astype(str).str.strip().str.lower()
        df[label_column] = df[label_column].astype(str).str.strip()

        # 2. Balance the Dataset (Oversampling)
        # This duplicates "Stress/Anxiety" rows so they equal "Normal" count
        max_size = df[label_column].value_counts().max()
        
        lst = []
        for class_index, group in df.groupby(label_column):
            # Resample this group to match the size of the largest group
            lst.append(group.sample(max_size, replace=True, random_state=42))
        
        df_balanced = pd.concat(lst)
        
        # Shuffle the data
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        print("Original Class Distribution:\n", df[label_column].value_counts())
        print("Balanced Class Distribution:\n", df_balanced[label_column].value_counts())

        # Store unique labels
        self.unique_labels = sorted(df_balanced[label_column].unique().tolist())
        
        texts = df_balanced[text_column].tolist()
        labels = df_balanced[label_column].tolist()

        return texts, labels

    def train(
        self,
        csv_path: str,
        text_column: str = "Text",
        label_column: str = "Mental_Health_Status",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        """Train model on provided dataset."""
        print("Loading and cleaning dataset...")
        texts, labels_raw = self.load_and_clean_dataset(csv_path, text_column, label_column)

        print(f"Total samples: {len(texts)}")
        print(f"Unique labels: {self.unique_labels}")

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.unique_labels)
        labels_encoded = self.label_encoder.transform(labels_raw)

        # Train-test split
        texts_train, texts_val, labels_train, labels_val = train_test_split(
            texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )

        print(f"Training samples: {len(texts_train)}, Validation samples: {len(texts_val)}")

        # Initialize tokenizer and model
        print("Initializing model...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(self.unique_labels)
        )

        # Create datasets
        train_dataset = TextDataset(texts_train, labels_train, self.tokenizer)
        val_dataset = TextDataset(texts_val, labels_val, self.tokenizer)

        # Training arguments
        # Use minimal TrainingArguments for compatibility across transformers versions
        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "training_output"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        print("Training model...")
        trainer.train()
        print("Training complete!")

        # Save model and tokenizer
        self.save_model()

    def save_model(self):
        """Save trained model, tokenizer, and label encoder."""
        print(f"Saving model to {self.model_dir}...")
        self.model.save_pretrained(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))

        # Save label encoder and unique labels
        with open(self.model_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(self.model_dir / "labels.json", "w") as f:
            json.dump(self.unique_labels, f)

        print("Model saved successfully!")

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            "model_dir": str(self.model_dir),
            "num_labels": len(self.unique_labels),
            "labels": self.unique_labels,
        }


if __name__ == "__main__":
    # Example usage
    trainer = CustomTextModelTrainer(model_dir="./custom_text_model")

    # Train on your dataset
    trainer.train(
        csv_path="unique_mental_health_dataset_26k_updated_disgust.csv",
        text_column="Text",
        label_column="Mental_Health_Status",
        epochs=3,
        batch_size=16,
    )

    print("Model training complete!")
    print(f"Model info: {trainer.get_model_info()}")
