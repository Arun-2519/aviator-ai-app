import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Aviator Strategy Assistant", layout="centered")
st.title("🕹️ Aviator Game Vision + Strategy AI Assistant")

st.sidebar.info("Upload a screenshot AND/OR enter last 10+ crash multipliers manually, then receive betting advice. App makes predictions once 10 or more rounds are provided.")

# Session state for crash history and confidence threshold
if 'crash_history' not in st.session_state:
    st.session_state.crash_history = []
st.session_state.conf_threshold = st.sidebar.slider("Confidence threshold (%)", 50, 100, 65)

# --- Helper functions
def extract_multiplier_from_text(text):
    """Parse multipliers from user-inputted text numbers ('1.31 2.01 74.11 ...')."""
    vals = []
    for t in text.replace(',', ' ').replace('x', ' ').split():
        try:
            v = float(t.strip())
            if 0.9 < v < 100:
                vals.append(v)
        except Exception:
            continue
    return vals

def predict_bet_advice(history, confidence_threshold):
    """Simple rules based on last 10 crash values."""
    if len(history) < 10:
        return "❌ Wait", None, 50, f"Not enough data ({len(history)} rounds). Need at least 10 rounds for prediction."
    # Analyze last 10 only
    recent = history[-10:]
    low_crashes = [x for x in recent if x < 1.5]
    recent_high = any(x > 4 for x in recent[-3:])
    volatility = len(low_crashes) > 3
    confidence = 90 if not volatility and not recent_high else 60
    if confidence < confidence_threshold:
        return "❌ Wait", None, confidence, "High volatility or pattern unstable, best to wait."
    if recent_high:
        return "❌ Wait", None, confidence, "Wait after a recent high crash to avoid risk."
    recommended_cashout = 2.0 if volatility else 1.7
    return "✅ Bet now", recommended_cashout, confidence, "Pattern semi-stable. Aim for a cautious cashout."

def show_predictions():
    h = st.session_state.crash_history
    advice, cashout, conf, note = predict_bet_advice(h, st.session_state.conf_threshold)
    st.subheader("Prediction Results")
    st.markdown(f"""
- **🎯 Bet Advice:** {advice}
- **💰 Cash Out Recommendation:** {cashout or '--'}x  
- **📊 Confidence:** {conf}%  
- **📢 Instruction:** {note}
    """)
    st.markdown("*Crash History (last 10 rounds):* " + ', '.join(f"{c}x" for c in h[-10:]))

# --- Input & Display Section

uploaded_file = st.file_uploader("Upload Aviator Game Screenshot (Image, PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Screenshot")
    st.info("Please ALSO input the crash multipliers visible in your screenshot (or your last 10+ crash values) for prediction.")

multipliers_text = st.text_area(
    "Enter last 10+ crash multipliers (from oldest to newest, separated by space/comma/x):",
    placeholder="e.g. 1.34x 2.15x 39.10x 1.05x ..."
)
if multipliers_text:
    vals = extract_multiplier_from_text(multipliers_text)
    if vals:
        st.session_state.crash_history = vals[-20:]  # Keep last up to 20 for efficiency
        show_predictions()
    else:
        st.warning("Please enter at least 10 valid multipliers.")

# Reset/copy function
if st.button("Reset Data / Clear History"):
    st.session_state.crash_history = []
    st.success("Data cleared! Ready for a new session.")
