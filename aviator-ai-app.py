import streamlit as st
from PIL import Image
import numpy as np
import easyocr

st.set_page_config(page_title="Aviator Strategy Assistant", layout="centered")
st.title("🕹️ Aviator Game Vision + Strategy AI Assistant")

st.sidebar.info("Upload a screenshot from the Aviator game. The system will extract crash multipliers and make a prediction once at least 10 are detected.")

st.session_state.conf_threshold = st.sidebar.slider("Confidence threshold (%)", 50, 100, 65)

# OCR Setup (uses CPU/gpu=False for compatibility)
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)
ocr_reader = get_ocr_reader()

def extract_multipliers_from_image(image):
    results = ocr_reader.readtext(np.array(image))
    multipliers = []
    for _, text, _ in results:
        text = text.strip().replace('x', '').replace('X', '').replace(',', '.')
        try:
            val = float(text)
            if 0.9 < val < 100:
                multipliers.append(val)
        except Exception:
            continue
    return multipliers

def predict_bet_advice(history, confidence_threshold):
    if len(history) < 10:
        return "❌ Wait", None, 50, f"Not enough data ({len(history)} found). Need at least 10 rounds for prediction."
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

def show_predictions(history):
    advice, cashout, conf, note = predict_bet_advice(history, st.session_state.conf_threshold)
    st.subheader("Prediction Results")
    st.markdown(f"""
- **🎯 Bet Advice:** {advice}
- **💰 Cash Out Recommendation:** {cashout or '--'}x  
- **📊 Confidence:** {conf}%  
- **📢 Instruction:** {note}
    """)
    st.markdown("*Crash History (last 10 rounds):* " + ', '.join(f"{c}x" for c in history[-10:]))

uploaded_file = st.file_uploader("Upload a clear Aviator Game Screenshot (PNG/JPG/JPEG) with crash multipliers visible", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Screenshot")
    multipliers = extract_multipliers_from_image(image)
    if len(multipliers) == 0:
        st.error("No numeric crash multipliers detected in screenshot. Try a clearer image.")
    else:
        st.success(f"Detected multipliers: {', '.join(str(m) + 'x' for m in multipliers)}")
        show_predictions(multipliers)

if st.button("Clear All"):
    st.experimental_rerun()
