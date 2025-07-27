import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import pandas as pd

st.set_page_config(page_title="Aviator Strategy Assistant", layout="centered")
st.title("🕹️ Aviator Game Vision + Strategy AI Assistant")

st.sidebar.info(
    "1. Upload your initial Aviator screenshot (the model learns from all detected multipliers, newest at top).\n"
    "2. Enter each new round's multiplier manually after each real round and get new predictions.\n"
    "3. All round data is stored and available for download as CSV."
)

conf_threshold = st.sidebar.slider("Confidence threshold (%)", 50, 100, 65)

# OCR Setup
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)
ocr_reader = get_ocr_reader()

if 'crash_history' not in st.session_state:
    st.session_state.crash_history = []

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
    recent = history[:10]  # newest 10
    low_crashes = [x for x in recent if x < 1.5]
    recent_high = any(x > 4 for x in recent[:3])
    volatility = len(low_crashes) > 3
    confidence = 90 if not volatility and not recent_high else 60
    if confidence < confidence_threshold:
        return "❌ Wait", None, confidence, "High volatility or pattern unstable, best to wait."
    if recent_high:
        return "❌ Wait", None, confidence, "Wait after a recent high crash to avoid risk."
    recommended_cashout = 2.0 if volatility else 1.7
    return "✅ Bet now", recommended_cashout, confidence, "Pattern semi-stable. Aim for a cautious cashout."

def show_predictions(history):
    advice, cashout, conf, note = predict_bet_advice(history, conf_threshold)
    st.subheader("Prediction Results")
    st.markdown(f"""
- **🎯 Bet Advice:** {advice}
- **💰 Cash Out Recommendation:** {cashout or '--'}x  
- **📊 Confidence:** {conf}%  
- **📢 Instruction:** {note}
    """)
    st.markdown("*Crash History (last 10, newest first):* " + ', '.join(f"{c}x" for c in history[:10]))

    # Show all stored data
    with st.expander("Show all stored crash data"):
        df = pd.DataFrame({'Crash Multiplier (newest first)': history})
        st.dataframe(df, use_container_width=True)

    # Save as CSV for download with unique key
    csv = pd.DataFrame({'Crash Multiplier': history})
    st.download_button(
        "Download All History as CSV",
        csv.to_csv(index=False),
        file_name="aviator_crash_history.csv",
        key=f"download_button_{len(history)}"
    )

uploaded_file = st.file_uploader(
    "Step 1: Upload a clear Aviator Game Screenshot (PNG/JPG/JPEG) with crash multipliers visible (newest at top)", 
    type=["png", "jpg", "jpeg"])

# Step 1: Only allow screenshot upload if user has not already started a session
if uploaded_file and not st.session_state.crash_history:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Screenshot")
    multipliers = extract_multipliers_from_image(image)
    if len(multipliers) == 0:
        st.error("No numeric crash multipliers detected in screenshot. Try a clearer image.")
    else:
        st.session_state.crash_history = multipliers
        st.success(f"Detected (newest first): {', '.join(str(m) + 'x' for m in multipliers)}")
        show_predictions(st.session_state.crash_history)

# Step 2: After screenshot, always allow user to add next result manually for continuous prediction
if len(st.session_state.crash_history) > 0:
    st.markdown("### Step 2: After each new Aviator round, enter the next crash multiplier below and get updated advice.")
    with st.form("add_next_multiplier_form"):
        next_multiplier = st.text_input(
            "Enter the NEXT round's crash multiplier (e.g. 2.81 or 1.09):"
        )
        submitted = st.form_submit_button("Add Result & Update Prediction")
    if submitted:
        try:
            val = float(next_multiplier.strip().replace('x', '').replace('X', '').replace(',', '.'))
            if 0.9 < val < 100:
                st.session_state.crash_history.insert(0, val)
                st.success(f"Added {val}x as the newest round. Prediction updated below.")
            else:
                st.warning("Please enter a valid multiplier (greater than 0.9 and less than 100).")
        except:
            st.warning("Input not recognized as a valid multiplier.")

    show_predictions(st.session_state.crash_history)

if st.button("Clear All Data"):
    st.session_state.crash_history = []
    st.success("All session data cleared.")
