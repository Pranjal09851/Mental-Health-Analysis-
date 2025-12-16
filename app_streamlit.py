"""
Streamlit app for custom text model training, prediction, and dual-modal mental health analysis.
"""

import os
import json
import tempfile
import streamlit as st
from datetime import datetime
from pathlib import Path
import pandas as pd

from fusion_utils import load_text_model, load_speech_model, predict_text, predict_speech, fuse

# Model paths
TEXT_MODEL_PATH = Path(os.getenv("TEXT_MODEL_PATH", "./custom_text_model")).resolve()
SPEECH_MODEL_PATH = Path(os.getenv("SPEECH_MODEL_PATH", "./speech_model")).resolve()

st.set_page_config(page_title="Mental-Health Assistant & Model Trainer", layout="wide")

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root {
    --blue: #4A90E2;
    --success: #4CAF50;
    --warning: #FFA654;
    --danger: #FF4B4B;
    --gray: #F5F6FA;
}
body, html { font-family: 'Inter', sans-serif; }
.header { 
    background: linear-gradient(90deg, var(--blue), #6fb3ff); 
    padding: 30px; 
    border-radius: 12px; 
    color: white; 
    margin-bottom: 20px;
}
.header h1 { margin: 0; font-size: 28px; }
.header p { margin: 5px 0 0 0; opacity: 0.9; }
.card { 
    background: white; 
    padding: 20px; 
    border-radius: 10px; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
    margin-bottom: 15px;
}
.result-box {
    background: var(--gray);
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid var(--blue);
}
.label-badge {
    display: inline-block;
    background: var(--blue);
    color: white;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 600;
    margin: 5px 0;
}
.success-badge { background: var(--success); }
.warning-badge { background: var(--warning); color: black; }
.danger-badge { background: var(--danger); }
.confidence { font-size: 18px; font-weight: 700; }
.muted { color: #6b7280; font-size: 13px; }
.small-muted { font-size: 12px; color: #777; }
.card-title { font-weight: 700; font-size: 16px; margin-bottom: 8px; }
.status-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
.badge { padding: 8px 12px; border-radius: 999px; font-weight: 700; }
.chat-box { max-height: 360px; overflow: auto; padding: 8px; }
.chat-user { background: #f1f5f9; padding: 10px 14px; border-radius: 14px; margin: 6px 0; display: inline-block; }
.chat-ai { background: linear-gradient(180deg, rgba(217,239,255,0.6), rgba(232,230,255,0.6)); padding: 10px 14px; border-radius: 14px; margin: 6px 0; display: inline-block; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def model_status_dot(available: bool):
    color = "--success" if available else "--danger"
    return f"<span class='status-dot' style='background:var({color})'></span>"


def init_models(text_path=str(TEXT_MODEL_PATH), speech_path=str(SPEECH_MODEL_PATH)):
    """Load models with error handling."""
    text_bundle = None
    speech_bundle = None
    text_error = None
    speech_error = None
    try:
        text_bundle = load_text_model(text_path)
    except Exception as e:
        text_error = str(e)
    try:
        speech_bundle = load_speech_model(speech_path)
    except Exception as e:
        speech_error = str(e)
    return text_bundle, speech_bundle, text_error, speech_error


def append_chat(role, content):
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    st.session_state["chat"].append({"role": role, "content": content})


def render_badge(risk: str):
    color = "var(--danger)"
    if risk.lower().startswith("moderate"):
        color = "var(--warning)"
    elif risk.lower().startswith("low"):
        color = "var(--success)"
    return f"<span class='badge' style='background:{color}; color: #fff'>{risk}</span>"


def donut_html(percent: int, color: str):
    return f"""
    <div style='display:flex;align-items:center;gap:12px'>
      <div style='width:110px;height:110px;border-radius:50%;background:conic-gradient({color} {percent}%, #eee {percent}%);display:grid;place-items:center'>
         <div style='font-weight:700'>{percent}%</div>
      </div>
    </div>
    """


# ========== DUAL-MODAL PAGE ==========

def page_dual_modal():
    """Dual-modal text + speech fusion analysis."""
    st.markdown("<div class='header'><h1>🎯 Dual-Modal Mental-Health Assistant</h1><p>AI-powered assessment using text + voice signals</p></div>", unsafe_allow_html=True)
    
    text_bundle, speech_bundle, text_err, speech_err = init_models()

    # Mode Selection
    mode = st.radio("Select Analysis Mode", ["Text Only", "Speech Only", "Dual Modal"], index=2, horizontal=True)
    st.markdown("---")

    # Input area
    # Define text input block
    def render_text_input():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📝 Text Input</div>", unsafe_allow_html=True)
        if "text_input" not in st.session_state:
            st.session_state["text_input"] = ""
            
        txt_input_val = st.text_area(
            "", 
            value=st.session_state["text_input"], 
            placeholder="Describe what you're feeling...", 
            key="text_input", 
            height=150
        )
        st.markdown(f"<div class='small-muted'>{len(txt_input_val)} characters</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return txt_input_val

    # Define speech input block
    def render_speech_input():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎙️ Audio Input</div>", unsafe_allow_html=True)
        audio_file = st.file_uploader("Drop or select audio file", type=["wav", "mp3", "m4a", "ogg"])
        if audio_file:
            st.markdown(f"<div class='small-muted'>✅ {audio_file.name}</div>", unsafe_allow_html=True)
            st.audio(audio_file)
        else:
            st.markdown("<div class='muted'>No audio selected</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return audio_file

    txt_input_val = ""
    audio_file = None

    if mode == "Dual Modal":
        col1, col2 = st.columns([1, 1])
        with col1:
            txt_input_val = render_text_input()
        with col2:
            audio_file = render_speech_input()
    elif mode == "Text Only":
        # Centered layout for single mode
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            txt_input_val = render_text_input()
    elif mode == "Speech Only":
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            audio_file = render_speech_input()

    # Analyze button
    st.markdown("<div style='text-align:center;margin:20px 0'>", unsafe_allow_html=True)
    
    # We allow running if the USER typed something (txt_input_val) OR if they uploaded a file.
    can_run = (len(txt_input_val) > 0) or (audio_file is not None)
    
    if st.button(
        "🔍 Analyze Mental Health", 
        disabled=not can_run
    ):
        with st.spinner("Analyzing..."):
            text_pred = None
            speech_pred = None
            
            # --- Use the DIRECT text input (txt_input_val) for analysis ---
            submitted_txt = txt_input_val
            
            if len(submitted_txt) > 0 and text_bundle:
                try:
                    text_pred = predict_text(submitted_txt, text_bundle)
                except Exception as e:
                    text_pred = {"label": "error", "confidence": 0.0, "error": str(e)}
            
            if audio_file and speech_bundle:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = tmp.name
                try:
                    speech_pred = predict_speech(tmp_path, speech_bundle)
                except Exception as e:
                    speech_pred = {"label": "error", "confidence": 0.0, "error": str(e)}
                finally:
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
            
            result = fuse(text_pred or {"label": "unknown", "confidence": 0.0}, speech_pred)
            
            # Log the interaction using the SUBMITTED text
            if len(submitted_txt) > 0:
                append_chat("user", {"text": submitted_txt})
            elif audio_file:
                 append_chat("user", {"text": "[Audio Analysis Only]"})
                 
            append_chat("text", text_pred or {})
            append_chat("speech", speech_pred or {})
            append_chat("result", result)

    st.markdown("</div>", unsafe_allow_html=True)

    # Results
    # ... (Rest of the function remains the same) ...
        # Results
    if "chat" in st.session_state and st.session_state["chat"]:
        out1, out2, out3 = st.columns(3)

        # ---------- Final Risk ----------
        with out1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📊 Final Risk</div>", unsafe_allow_html=True)

            result_msg = [m for m in st.session_state["chat"] if m["role"] == "result"]
            if result_msg:
                res = result_msg[-1]["content"]
                risk = res.get("final_risk", "Unknown")

                # Donut color logic
                risk_lower = risk.lower()
                if "high" in risk_lower:
                    pct = 95
                    color = "var(--danger)"
                elif "moderate" in risk_lower:
                    pct = 65
                    color = "var(--warning)"
                else:
                    pct = 25
                    color = "var(--success)"

                st.markdown(render_badge(risk), unsafe_allow_html=True)
                st.markdown(donut_html(pct, color), unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Text Result ----------
        with out2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📝 Text Result</div>", unsafe_allow_html=True)

            text_msg = [m for m in st.session_state["chat"] if m["role"] == "text"]
            if text_msg:
                t = text_msg[-1]["content"]
                st.write(f"**{t.get('label', '-')}**")
                st.write(f"Confidence: {t.get('confidence', 0):.2%}")

                # Add meditation suggestion
                if t.get("recommendation"):
                    st.markdown(f"**Recommendation:** {t['recommendation']}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Speech Result ----------
        with out3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🎙️ Speech Result</div>", unsafe_allow_html=True)

            speech_msg = [m for m in st.session_state["chat"] if m["role"] == "speech"]
            if speech_msg:
                s = speech_msg[-1]["content"]
                st.write(f"**{s.get('label', '-')}**")
                st.write(f"Confidence: {s.get('confidence', 0):.2%}")

            st.markdown("</div>", unsafe_allow_html=True)


# ========== MAIN APP ==========

if __name__ == "__main__":
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='card'><h3>Model Status</h3>", unsafe_allow_html=True)
        text_bundle, speech_bundle, text_err, speech_err = init_models()
        st.markdown(f"{model_status_dot(text_bundle is not None)} Text model", unsafe_allow_html=True)
        st.markdown(f"{model_status_dot(speech_bundle is not None)} Speech model", unsafe_allow_html=True)
        
        if text_err:
            with st.expander("Text error"):
                st.code(text_err)
        if speech_err:
            with st.expander("Speech error"):
                st.code(speech_err)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Run the app
    page_dual_modal()