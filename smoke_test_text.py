"""Simple smoke test to load the DistilBERT text model and run one prediction.

Run inside your activated venv from the project folder:
    python smoke_test_text.py
"""
import traceback
from pathlib import Path

MODEL_PATH = Path("./custom_text_model")

def main():
    print("Smoke test: load text model from", MODEL_PATH.resolve())
    try:
        from fusion_utils import load_text_model, predict_text
    except Exception as e:
        print("Failed to import fusion_utils:")
        traceback.print_exc()
        return

    try:
        bundle = load_text_model(str(MODEL_PATH))
        print("Loaded text model bundle. Tokenizer and model present.")
    except Exception as e:
        print("Failed to load text model:")
        traceback.print_exc()
        return

    try:
        sample = "I don't want to live anymore"
        print("Predicting sample text:", sample)
        pred = predict_text(sample, bundle)
        print("Prediction:", pred)
    except Exception as e:
        print("Prediction failed:")
        traceback.print_exc()


if __name__ == '__main__':
    main()
