import os
import joblib
import numpy as np
import json
import pickle

try:
    # transformers for text model
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    HAS_TF = False

# --- ADDED: Recommendations Mapping ---
LABEL_RECOMMENDATIONS = {
    "depression": "🧘 Meditation is recommended. Try a 5-10 minute guided meditation to calm your mind.",
    "anxiety": "🧘 Meditation is recommended. Deep breathing exercises can help reduce anxiety.",
    "stress": "🧘 Meditation is recommended. A mindfulness session may help alleviate stress.",
}

def load_text_model(path="/kaggle/working/text_model_final"):
    # ... function body remains the same ...
    # ...
    """Load tokenizer, model and label encoder saved at `path`."""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers or torch not available in environment")

    last_exc = None
    # Try local folder first
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(path, local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(path, local_files_only=True)
    except Exception as e:
        last_exc = e
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(path, local_files_only=False)
            model = DistilBertForSequenceClassification.from_pretrained(path, local_files_only=False)
            last_exc = None
        except Exception as e2:
            last_exc = e2

    if last_exc is not None:
        hf_ref = os.getenv("TEXT_MODEL_REF")
        if hf_ref:
            try:
                tokenizer = DistilBertTokenizerFast.from_pretrained(hf_ref, local_files_only=False)
                model = DistilBertForSequenceClassification.from_pretrained(hf_ref, local_files_only=False)
                last_exc = None
            except Exception as e3:
                last_exc = e3

    if last_exc is not None:
        raise RuntimeError(f"Failed to load text model from '{path}'. Original error: {last_exc}")

    le = None
    labels = None

    le_path = os.path.join(path, "label_encoder.joblib")
    if os.path.exists(le_path):
        try:
            le = joblib.load(le_path)
        except Exception:
            le = None

    if le is None:
        pkl_path = os.path.join(path, "label_encoder.pkl")
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    le = pickle.load(f)
            except Exception:
                le = None

    labels_path = os.path.join(path, "labels.json")
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = None

    return {"tokenizer": tokenizer, "model": model, "le": le, "labels": labels}


def predict_text(text, text_model_bundle):
    """Return predicted label string and probability for input text."""
    tokenizer = text_model_bundle["tokenizer"]
    model = text_model_bundle["model"]
    le = text_model_bundle.get("le")
    labels = text_model_bundle.get("labels")

    # --- FIX 1: Match Training Preprocessing (Lowercase + Strip) ---
    text = str(text).strip().lower()

    model.eval()
    inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    # Determine Label
    if labels:
        try:
            label = str(labels[int(pred_idx)])
        except Exception:
            label = str(pred_idx)
    elif le is not None:
        try:
            label = le.inverse_transform([pred_idx])[0]
        except Exception:
            if hasattr(model.config, "id2label"):
                label = model.config.id2label.get(pred_idx, str(pred_idx))
            else:
                label = str(pred_idx)
    else:
        if hasattr(model.config, "id2label"):
            label = model.config.id2label.get(pred_idx, str(pred_idx))
        else:
            label = str(pred_idx)

    # Build probability distribution mapping
    prob_dist = {}
    if labels:
        for i, lab in enumerate(labels):
            prob_dist[str(lab)] = float(probs[i])
    else:
        if hasattr(model.config, "id2label"):
            for i in range(len(probs)):
                prob_dist[model.config.id2label.get(i, str(i))] = float(probs[i])
        else:
            for i in range(len(probs)):
                prob_dist[str(i)] = float(probs[i])

    confidence = float(probs[pred_idx])

    # --- FIX 2: Add Recommendation Logic ---
    recommendation = None
    pred_label_lower = str(label).lower()
    if pred_label_lower in LABEL_RECOMMENDATIONS:
        recommendation = LABEL_RECOMMENDATIONS[pred_label_lower]

    return {
        "label": str(label), 
        "confidence": confidence, 
        "probabilities": prob_dist,
        "recommendation": recommendation  # Now included in return!
    }


def load_speech_model(path="/kaggle/working/speech_model"):
    """Load speech model, scaler and label encoder if present."""
    model = None
    scaler = None
    le = None

    if HAS_TF:
        try:
            model_file = os.path.join(path, "speech_emotion_model.h5")
            if os.path.exists(model_file):
                model = tf.keras.models.load_model(model_file)
            else:
                model = tf.keras.models.load_model(path)
        except Exception:
            model = None

    if model is None:
        pkl = os.path.join(path, "audio_model.pkl")
        if os.path.exists(pkl):
            try:
                model = joblib.load(pkl)
            except Exception:
                model = None

    scaler_path = os.path.join(path, "speech_scaler.joblib")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None
    else:
        alt = os.path.join(path, "audio_scaler.pkl")
        if os.path.exists(alt):
            try:
                scaler = joblib.load(alt)
            except Exception:
                scaler = None

    le_path = os.path.join(path, "speech_label_encoder.joblib")
    if os.path.exists(le_path):
        try:
            le = joblib.load(le_path)
        except Exception:
            le = None
    else:
        alt = os.path.join(path, "audio_label_encoder.pkl")
        if os.path.exists(alt):
            try:
                le = joblib.load(alt)
            except Exception:
                le = None

    return {"model": model, "scaler": scaler, "le": le}


def extract_speech_features(audio_path, sr=22050, n_mfcc=40, target_mel_len=128, duration=3.0):
    try:
        import librosa
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        if y is None or len(y) == 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel)
        mel_mean = np.mean(mel_db, axis=1)

        if mel_mean.shape[0] < target_mel_len:
            mel_mean = np.pad(mel_mean, (0, target_mel_len - mel_mean.shape[0]), mode="constant")
        else:
            mel_mean = mel_mean[:target_mel_len]

        feat = np.concatenate([mfcc_mean, mel_mean])
        return feat
    except Exception:
        return None


def predict_speech(audio_path, speech_bundle):
    bundle_model = speech_bundle.get("model")
    scaler = speech_bundle.get("scaler")
    le = speech_bundle.get("le")

    feat = extract_speech_features(audio_path)
    if feat is None:
        return None

    X = feat.reshape(1, -1)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    if bundle_model is None:
        raise RuntimeError("Speech model not loaded")

    try:
        if HAS_TF and hasattr(bundle_model, "predict"):
            probs = bundle_model.predict(X)
            if probs.ndim > 1:
                probs = probs[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
        else:
            probs = bundle_model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
    except Exception:
        try:
            pred = bundle_model.predict(X)
            pred_idx = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            conf = 1.0
        except Exception as e:
            raise

    label = None
    if le is not None:
        try:
            label = le.inverse_transform([pred_idx])[0]
        except Exception:
            pass

    if label is None:
        if hasattr(bundle_model, "classes_"):
            try:
                label = str(bundle_model.classes_[pred_idx])
            except Exception:
                label = str(pred_idx)
        else:
            label = str(pred_idx)

    return {"label": str(label), "confidence": float(conf)}


def fuse(text_prediction, speech_prediction):
    label = text_prediction.get("label", "").lower()
    confidence = float(text_prediction.get("confidence", 0))

    if label == "depression":
        if confidence >= 0.90:
            risk = "High Risk"
        elif confidence >= 0.70:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

    elif label == "anxiety":
        if confidence >= 0.80:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

    elif label == "stress":
        if confidence >= 0.80:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

    else:
        risk = "Low Risk"

    speech_label = None
    if speech_prediction:
        speech_label = str(speech_prediction.get("label", "")).lower()

    negative_emotions = ["sad", "sadness", "fear", "angry"]

    if speech_label in negative_emotions:
        if risk == "Low Risk":
            risk = "Moderate Risk"
        elif risk == "Moderate Risk":
            risk = "High Risk"

    return {
        "final_risk": risk,
        "explanation": f"Text: {label} ({confidence:.2f}), Audio: {speech_label}",
        "text": text_prediction,
        "speech": speech_prediction
    }