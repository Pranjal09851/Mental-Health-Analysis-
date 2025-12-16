"""
Load and run predictions with the custom trained text model.
Includes label→recommendation mapping (Depression, Anxiety, Stress → Meditation).
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Label to Recommendation Mapping
LABEL_RECOMMENDATIONS = {
    "depression": "🧘 Meditation is recommended. Try a 5-10 minute guided meditation to calm your mind.",
    "anxiety": "🧘 Meditation is recommended. Deep breathing exercises can help reduce anxiety.",
    "stress": "🧘 Meditation is recommended. A mindfulness session may help alleviate stress.",
}


class CustomTextModelPredictor:
    """Load and run predictions with trained custom model."""

    def __init__(self, model_dir: str = "./custom_text_model"):
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.unique_labels = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        # Only load if directory exists
        if self.model_dir.exists():
            self.load_model()
        else:
            print(f"⚠️  Model directory not found: {self.model_dir}")
            print("Train a model first using train_model.py")

    def load_model(self):
        """Load tokenizer, model, and label encoder from disk."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        print(f"Loading model from {self.model_dir}...")

        # Load tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(self.model_dir))

        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        # Load label encoder and labels
        with open(self.model_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(self.model_dir / "labels.json", "r") as f:
            self.unique_labels = json.load(f)

        self.model_loaded = True
        print(f"✅ Model loaded. Labels: {self.unique_labels}")

    def predict(
        self, text: str
    ) -> Dict:
        """
        Run prediction on input text.

        Returns dict with:
        - label: predicted class label
        - confidence: confidence score (0-1)
        - probabilities: prob for each class
        - recommendation: optional meditation message if applicable
        """
        if not self.model_loaded or not self.tokenizer or not self.model:
            raise RuntimeError("❌ Model not loaded. Train a model first using train_model.py")

        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Run model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))

        # Get label and confidence
        pred_label = self.unique_labels[pred_idx]
        confidence = float(probs[pred_idx])

        # Build probability distribution
        prob_dist = {label: float(probs[i]) for i, label in enumerate(self.unique_labels)}

        # Check if recommendation applies
        recommendation = None
        pred_label_lower = pred_label.lower()
        if pred_label_lower in LABEL_RECOMMENDATIONS:
            recommendation = LABEL_RECOMMENDATIONS[pred_label_lower]

        return {
    "label": pred_label,
    "confidence": confidence,
    "probabilities": prob_dist,
    "recommendation": recommendation or "You're doing okay. Keep taking care of yourself."
}


    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """Run predictions on multiple texts."""
        return [self.predict(text) for text in texts]

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            "model_dir": str(self.model_dir),
            "num_labels": len(self.unique_labels),
            "labels": self.unique_labels,
            "device": str(self.device),
        }


if __name__ == "__main__":
    # Example: load model and run a prediction
    predictor = CustomTextModelPredictor(model_dir="./custom_text_model")

    sample_text = "I'm feeling really sad and hopeless lately"
    result = predictor.predict(sample_text)

    print(f"\nPrediction for: '{sample_text}'")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Probabilities: {result['probabilities']}")
    if result['recommendation']:
        print(f"Recommendation: {result['recommendation']}")
