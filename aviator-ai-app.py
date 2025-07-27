import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import easyocr

st.set_page_config(page_title="Aviator Strategy Assistant", layout="centered")
st.title("🕹️ Aviator Game Vision + Strategy AI Assistant")

st.sidebar.info(
    "1. Upload your Aviator screenshot (extracts all crash multipliers, newest at top).\n"
    "2. After each round, enter the new multiplier; the model learns color streaks and gives advice.\n"
    "3. Download FULL history anytime."
)

conf_threshold = st.sidebar.slider("Confidence threshold (%)", 50, 100, 65)

@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)
ocr_reader = get_ocr_reader()

if 'crash_history' not in st.session_state:
    st.session_state.crash_history = []
if 'color_history' not in st.session_state:
    st.session_state.color_history = []

def multiplier_to_color(mult):
    if 1 <= mult < 2:
        return "blue"
    elif 2 <= mult < 10:
        return "dark blue"
    elif mult >= 10:
        return "pink"
    else:
        return "unknown"

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

def analyze_color_streak(colors):
    """Analyze streaks and alternation for smarter AI advice."""
    if not colors:
        return 0, "", 0, "", 0
    # Count current streak (first color repeated at front)
    streak_color = colors[0]
    streak_len = 1
    for c in colors[1:]:
        if c == streak_color:
            streak_len += 1
        else:
            break
    # Count alternations within last 10
    alternations = 0
    for i in range(min(9, len(colors)-1)):
        if colors[i] != colors[i+1]:
            alternations += 1
    # Longest pink streak in last 10
    pink_streak = 0
    current_streak = 0
    for c in colors[:10]:
        if c == "pink":
            current_streak += 1
            if current_streak > pink_streak:
                pink_streak = current_streak
        else:
            current_streak = 0
    return streak_len, streak_color, alternations, colors[0], pink_streak

def predict_bet_advice(history, colors, confidence_threshold):
    if len(history) < 10:
        return "❌ Wait", None, 50, f"Not enough data ({len(history)} found). Need at least 10 rounds for prediction."
    recent = history[:10]
    color_recent = colors[:10]
    low_crashes = [x for x in recent if x < 1.5]
    volatility = len(low_crashes) > 3
    # Color streak analysis
    streak_len, streak_color, alternations, curr_color, pink_streak = analyze_color_streak(color_recent)
    confidence = 70
    note = []
    if curr_color == "pink":
        return "❌ Wait", None, 60, "Wait after a recent pink result (very high crash)."
    if streak_color == "blue" and streak_len >= 3:
        confidence = 92 if not volatility else 80
        note.append("Multiple consecutive blues: chance of bigger multiplier soon.")
    if pink_streak >= 2:
        confidence = 60
        note.append("Recent multiple pinks: avoid betting for a reset.")
    if alternations > 5:
        confidence = 60
        note.append("Rapid alternation (unstable, drop confidence).")
    if not note:
        note.append("Pattern semi-stable. Aim for a cautious cashout.")
    if confidence < confidence_threshold:
        return "❌ Wait", None, confidence, "Low model confidence due to detected volatility or pattern instability."
    recommended_cashout = 2.0 if volatility or (streak_color == "blue" and streak_len >= 3) else 1.7
    return "✅ Bet now", recommended_cashout, confidence, " ".join(note)

def show_predictions(history, colors):
    advice, cashout, conf, note = predict_bet_advice(history, colors, conf_threshold)
    st.subheader("Prediction Results")
    st.markdown(f"""
- **🎯 Bet Advice:** {advice}
- **💰 Cash Out Recommendation:** {cashout or '--'}x
- **📊 Confidence:** {conf}%
- **📢 Instruction:** {note}
""")
    st.markdown(
        "Crash History (last 10, newest first): " + " ".join(
            f'<span style="color:{"#457bba" if c=="blue" else "#042469" if c=="dark blue" else "#de0475"}"><b>{h}x</b></span>'
            for h, c in zip(history[:10], colors[:10])
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Show all crash data with color"):
        df = pd.DataFrame({
            'Crash Multiplier (newest first)': history,
            'Category': colors
        })
        st.dataframe(df, use_container_width=True)
    csv = pd.DataFrame({
        'Crash Multiplier': history,
        'Category': colors
    })
    st.download_button(
        "Download All History as CSV",
        csv.to_csv(index=False),
        file_name="aviator_crash_history.csv",
        key=f"download_button_{len(history)}"
    )

uploaded_file = st.file_uploader(
    "Step 1: Upload a clear Aviator Game Screenshot (PNG/JPG/JPEG) with crash multipliers visible (newest at top)", 
    type=["png", "jpg", "jpeg"])

if uploaded_file and not st.session_state.crash_history:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Screenshot")
    multipliers = extract_multipliers_from_image(image)
    if not multipliers:
        st.error("No numeric crash multipliers detected in screenshot. Try a clearer image.")
    else:
        colors = [multiplier_to_color(v) for v in multipliers]
        st.session_state.crash_history = multipliers
        st.session_state.color_history = colors
        st.success("Detected (newest first): " + ", ".join(f"{m}x [{c}]" for m, c in zip(multipliers, colors)))
        show_predictions(st.session_state.crash_history, st.session_state.color_history)

if st.session_state.crash_history:
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
                color = multiplier_to_color(val)
                st.session_state.crash_history.insert(0, val)
                st.session_state.color_history.insert(0, color)
                st.success(f"Added {val}x as the newest round ({color}). Prediction updated below.")
            else:
                st.warning("Please enter a valid multiplier (greater than 0.9 and less than 100).")
        except:
            st.warning("Input not recognized as a valid multiplier.")
    show_predictions(st.session_state.crash_history, st.session_state.color_history)

if st.button("Clear All Data"):
    st.session_state.crash_history = []
    st.session_state.color_history = []
    st.success("All session data cleared.")
