# Mental-Health-Analysis-
A dual-modal mental health analysis system that combines text emotion classification using DistilBERT and speech emotion recognition with MFCC features. The system fuses both modalities to assess psychological risk levels and provides wellness recommendations through a real-time Streamlit interface.

Key Features

Text Emotion Classification

Fine-tuned DistilBERT model

Detects emotions such as Depression, Anxiety, Stress, and Neutral states

Provides confidence scores and probability distributions

Speech Emotion Recognition

Extracts MFCC and mel-spectrogram features using Librosa

Supports ML (.pkl) and DL (.h5) speech models

Identifies vocal emotions like sadness, anger, fear, and neutrality

Multi-Modal Fusion Engine

Combines text and speech predictions

Generates final Low / Moderate / High risk assessment

Produces human-readable explanations

Recommendation System

Suggests meditation or wellness activities when emotional distress is detected

Interactive Web Interface

Built using Streamlit

Supports text input, audio upload, and real-time predictions

Displays confidence scores and risk interpretation

🏗 System Architecture

The system follows a late-fusion architecture:

Text input → DistilBERT emotion classifier

Audio input → Speech emotion recognition model

Fusion module → Risk scoring + recommendation

Streamlit UI → Visualization and interaction

This modular design allows independent model upgrades and easy system extension.

⚙️ Technologies Used

Python

Hugging Face Transformers (DistilBERT)

PyTorch

Librosa

Scikit-learn

TensorFlow/Keras (optional)

Streamlit

📁 Project Structure
├── app_streamlit.py          # Streamlit frontend
├── train_model.py            # Text model training script
├── custom_text_model.py      # Text prediction logic
├── fusion_utils.py           # Speech processing + fusion engine
├── run_predict.py            # Standalone prediction script
├── debug_load_models.py      # Model loading diagnostics
├── smoke_test_text.py        # Text model validation
├── smoke_test_speech.py      # Speech model validation
└── custom_text_model/        # Saved model artifacts

🚀 Use Cases

Mental wellness platforms

Early emotional distress screening

Academic research in affective computing

AI-based self-assessment tools

Multi-modal emotion analysis experiments

⚠️ Disclaimer

This system is not a medical diagnostic tool.
It is intended for educational, research, and wellness support purposes only and should not replace professional mental-health care.
