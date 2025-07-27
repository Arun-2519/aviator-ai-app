import streamlit as st
import cv2
import numpy as np
import easyocr
import time
from PIL import Image

st.set_page_config(page_title="Aviator Strategy Assistant", layout="centered")
st.title("🕹️ Aviator Game Vision + Strategy AI Assistant")

ocr = easyocr.Reader(['en'], gpu=False)

# Sidebar settings
bankroll = st.sidebar.number_input("Your bankroll (optional):", min_value=0)
confidence_threshold = st.sidebar.slider("Confidence threshold (%)", 50, 100, 65)
st.sidebar.info("Upload screenshots or use webcam. Model starts prediction only after 10 crash data points are collected.")

# Input source selection
input_source = st.radio("Choose Input Source:", ["Upload Screenshot(s)", "Webcam"])

# Global session state to keep crash history between frames
if 'crash_history' not in st.session_state:
    st.session_state.crash_history = []

if 'current_multiplier' not in st.session_state:
    st.session_state.current_multiplier = None


def extract_game_data(img):
    """
    Extract multipliers from image using OCR.
    Returns current multiplier and list of detected multipliers (crash history candidates).
    """
    res = ocr.readtext(img)
    text_list = [t[1] for t in res]
    multipliers = []
    for t in text_list:
        t = t.replace('x', '').replace('X', '').strip()
        # accept floats between 0.9 and 100 for sanity check
        try:
            val = float(t)
            if 0.9 < val < 100:
                multipliers.append(val)
        except Exception:
            continue
    if len(multipliers) == 0:
        return None, []
    # Last value is assumed current multiplier (live multiplier)
    current = multipliers[-1]
    history = multipliers[:-1]
    return current, history


def predict_bet_advice(history, curr_mult):
    """
    Use rules to decide bet or wait with confidence.
    """
    if len(history) < 10:
        return "❌ Wait", None, 50, f"Not enough data collected ({len(history)} rounds). Need 10 rounds before prediction."

    # Focus on last 10 rounds only to simulate "rolling" history
    recent = history[-10:]
    low_crashes = [x for x in recent if x < 1.5]
    recent_high = any(x > 4.0 for x in recent[-3:])
    volatility = len(low_crashes) > 3

    # Adjust confidence
    confidence = 90 if not volatility and not recent_high else 60

    # Respect confidence threshold from user input
    if confidence < confidence_threshold:
        return "❌ Wait", None, confidence, "High volatility or pattern unstable, best to wait."

    if recent_high:
        return "❌ Wait", None, confidence, "Waiting after recent high crash to avoid risk."

    # Recommend cashout multiplier
    recommended_cashout = 2.0 if volatility else 1.7
    return "✅ Bet now", recommended_cashout, confidence, "Pattern semi-stable. Aim for a cautious cashout."


def handle_new_data(image_np):
    curr, hist = extract_game_data(image_np)
    if curr is not None:
        st.session_state.current_multiplier = curr
    # Append newly detected crashes to global session history, avoiding duplicates and keeping length manageable
    # We'll assume session_state.crash_history holds confirmed past rounds only (not live multiplier)
    # Append only new unique values in hist that we don't already have to avoid duplicates
    new_rounds = [v for v in hist if v not in st.session_state.crash_history]
    st.session_state.crash_history.extend(new_rounds)
    # Keep last 20 for memory efficiency
    if len(st.session_state.crash_history) > 20:
        st.session_state.crash_history = st.session_state.crash_history[-20:]

    advice, cashout, confidence, instruction = predict_bet_advice(st.session_state.crash_history, st.session_state.current_multiplier)

    # Display results
    st.subheader("Prediction Results")
    st.markdown(f"""
- **🎯 Bet Advice:** {advice}  
- **💰 Cash Out Recommendation:** {cashout or '--'}x  
- **📊 Confidence:** {confidence}%  
- **📢 Instruction:** {instruction}
""")

    if len(st.session_state.crash_history) > 0:
        st.markdown("*Crash History (last 10 rounds):* " + ', '.join(f"{c}x" for c in st.session_state.crash_history[-10:]))

    if st.session_state.current_multiplier is not None:
        st.markdown(f"*Current Live Multiplier (if present):* {st.session_state.current_multiplier}x")

    return advice, cashout, confidence, instruction


if input_source == "Upload Screenshot(s)":
    uploaded_files = st.file_uploader(
        "Upload 1 or more Aviator game screenshots (upload multiple for accumulating data):",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Uploaded: {uploaded_file.name}")
            image_np = np.array(image)
            handle_new_data(image_np)

elif input_source == "Webcam":
    st.info("Show the Aviator game screen clearly to the webcam. The model will accumulate info and start predictions after 10 crash records.")

    start_cam = st.button("Start Webcam")
    stop_cam = st.button("Stop Webcam")

    if start_cam:
        cap = cv2.VideoCapture(0)
        frame_window = st.empty()
        count = 0
        max_frames = 60  # Limit run time ~1 minute or stop by user

        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to get frame from webcam.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, caption="Live Webcam Feed")

            # Process every 2 seconds (~60 fps -> every 120 frames), or more frequently if desired
            # For demo, process every loop iteration, but could be throttled:
            handle_new_data(frame_rgb)

            count += 1
            time.sleep(1)  # Adjust delay for CPU load and real-time processing

        cap.release()
        st.success("Webcam feed stopped. Reload page to restart.")

if st.button("Reset Data / Clear History"):
    st.session_state.crash_history = []
    st.session_state.current_multiplier = None
    st.success("Data cleared! Ready for new session.")
