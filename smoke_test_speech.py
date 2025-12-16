import tempfile
import wave
import struct
import os
import traceback

from fusion_utils import load_speech_model, predict_speech


def write_silent_wav(path, duration_s=1.0, sr=16000):
    n_frames = int(duration_s * sr)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        silence = struct.pack('<h', 0)
        for _ in range(n_frames):
            wf.writeframes(silence)


def main():
    speech_path = './speech_model'
    print('Smoke test: load speech model from', os.path.abspath(speech_path))
    try:
        bundle = load_speech_model(speech_path)
        if bundle.get('model') is None:
            print('Speech model bundle loaded but model is None')
            return
        print('Speech bundle loaded; attempting prediction on silent WAV...')
    except Exception as e:
        print('Failed to load speech model:')
        traceback.print_exc()
        return

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        write_silent_wav(tmp_path, duration_s=1.0)
        try:
            out = predict_speech(tmp_path, bundle)
            print('Prediction output:', out)
        except Exception as e:
            print('Prediction failed:')
            traceback.print_exc()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        print('Failed creating temp wav')
        traceback.print_exc()


if __name__ == '__main__':
    main()
"""Smoke test for speech model: tries common paths, loads model, writes a silent WAV and predicts."""
import tempfile
import wave
import struct
import os
from fusion_utils import load_speech_model, predict_speech

candidates = [
    "./speech_model",
    "./audio_model",
    "./outputs/speech_emotion_model.h5",
    "/kaggle/working/speech_model",
    "/kaggle/working/audio_model",
]

def write_silence(path, sr=16000, duration=1.0):
    n = int(sr * duration)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        silence = struct.pack('<h', 0)
        for _ in range(n):
            wf.writeframes(silence)


def main():
    for p in candidates:
        try:
            print('Trying:', p)
            bundle = load_speech_model(p)
            if bundle is None:
                print('-> bundle None')
                continue
            model = bundle.get('model')
            if model is None:
                print('-> no model loaded from', p)
                continue
            print('-> model loaded from', p, 'type=', type(model))

            fd, tmp = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            try:
                write_silence(tmp)
                try:
                    out = predict_speech(tmp, bundle)
                    print('-> predict output:', out)
                except Exception as e:
                    print('-> predict failed:', repr(e))
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception as e:
            print('-> load failed for', p, 'err:', repr(e))

    print('DONE')


if __name__ == '__main__':
    main()
