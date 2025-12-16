import os
import traceback
from pathlib import Path

from fusion_utils import load_text_model, load_speech_model

TEXT_MODEL_PATH = Path(os.getenv('TEXT_MODEL_PATH', './text_model_final'))
SPEECH_MODEL_PATH = Path(os.getenv('SPEECH_MODEL_PATH', './speech_model'))

def list_dir(p: Path):
    if not p.exists():
        print(f"Path not found: {p.resolve()}")
        return
    print(f"Listing: {p.resolve()}")
    for child in sorted(p.iterdir()):
        try:
            print(f" - {child.name} ({child.stat().st_size} bytes)")
        except Exception:
            print(f" - {child.name}")

def try_load_text():
    try:
        print('\nTrying to load text model from:', TEXT_MODEL_PATH)
        m = load_text_model(str(TEXT_MODEL_PATH))
        print('Text model loaded. Keys:', list(m.keys()))
    except Exception as e:
        print('Text model load FAILED:')
        traceback.print_exc()

def try_load_speech():
    try:
        print('\nTrying to load speech model from:', SPEECH_MODEL_PATH)
        m = load_speech_model(str(SPEECH_MODEL_PATH))
        print('Speech bundle loaded. Keys:', list(m.keys()))
    except Exception as e:
        print('Speech model load FAILED:')
        traceback.print_exc()

if __name__ == '__main__':
    print('Debug helper for model loading')
    list_dir(TEXT_MODEL_PATH)
    list_dir(SPEECH_MODEL_PATH)
    try_load_text()
    try_load_speech()
