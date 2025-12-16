# Simple runner for CustomTextModelPredictor
# Usage: python run_predict.py "Your text here"

import sys
from pathlib import Path

try:
    from custom_text_model import CustomTextModelPredictor
except Exception as e:
    print("Failed to import CustomTextModelPredictor:", e)
    sys.exit(1)

if __name__ == '__main__':
    text = "I feel anxious and stressed" if len(sys.argv) < 2 else sys.argv[1]
    predictor = CustomTextModelPredictor(model_dir="./custom_text_model")
    if not getattr(predictor, 'model_loaded', False):
        print("Model not loaded. Train a model first or place a trained model under ./custom_text_model/")
        sys.exit(0)

    try:
        res = predictor.predict(text)
        print("Input:", text)
        print("Label:", res['label'])
        print("Confidence:", f"{res['confidence']:.3f}")
        print("Probabilities:")
        for k, v in res['probabilities'].items():
            print(f"  {k}: {v:.3f}")
        if res.get('recommendation'):
            print("Recommendation:", res['recommendation'])
    except Exception as e:
        print("Prediction error:", e)
