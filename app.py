"""
╔══════════════════════════════════════════════════════════════╗
║         DIABETES PREDICTION SYSTEM — Streamlit App           ║
║         Dark Glassmorphism × Modern Health Dashboard         ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark Glassmorphism Theme
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Fonts ────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Orbitron:wght@600;700&display=swap');

/* ── Root Variables ──────────────────────────────────────── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #0d1426;
    --bg-card:       rgba(15, 25, 55, 0.75);
    --border-glass:  rgba(99, 179, 237, 0.18);
    --neon-blue:     #63b3ed;
    --neon-cyan:     #76e4f7;
    --neon-green:    #68d391;
    --neon-red:      #fc8181;
    --neon-orange:   #f6ad55;
    --text-primary:  #e2e8f0;
    --text-muted:    #718096;
    --text-accent:   #90cdf4;
    --shadow-neon:   0 0 20px rgba(99, 179, 237, 0.3);
    --shadow-card:   0 8px 32px rgba(0, 0, 0, 0.4);
    --radius-card:   16px;
    --radius-input:  10px;
}

/* ── Global Background ───────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1426 40%, #0f1932 70%, #0a0e1a 100%);
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Subtle animated grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(99,179,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,179,237,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.92) !important;
    border-right: 1px solid var(--border-glass);
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--neon-cyan) !important;
}

/* ── Main Header ─────────────────────────────────────────── */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    position: relative;
}

.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #76e4f7, #9f7aea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    letter-spacing: 2px;
    line-height: 1.1;
}

.main-subtitle {
    font-size: 1rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 300;
}

.header-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-blue), var(--neon-cyan), transparent);
    margin: 1.5rem auto;
    max-width: 600px;
}

/* ── Glass Card ──────────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-card);
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: var(--shadow-card);
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.6), transparent);
}

.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--neon-cyan);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Input Fields ────────────────────────────────────────── */
.stNumberInput input,
.stTextInput input,
.stSelectbox select {
    background: rgba(10, 20, 45, 0.8) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-input) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: var(--neon-blue) !important;
    box-shadow: 0 0 0 2px rgba(99, 179, 237, 0.2) !important;
    outline: none !important;
}

/* Input labels */
.stNumberInput label,
.stTextInput label,
.stCheckbox label {
    color: var(--text-accent) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Slider ──────────────────────────────────────────────── */
.stSlider [data-testid="stTickBar"] {
    background: none;
}
.stSlider .stSlider > div > div > div {
    background: var(--neon-blue) !important;
}

/* ── Predict Button ──────────────────────────────────────── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1e40af, #2563eb, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.85rem 3rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.5), 0 0 40px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    width: 100% !important;
}

div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6, #38bdf8) !important;
    box-shadow: 0 6px 30px rgba(59, 130, 246, 0.7), 0 0 60px rgba(59, 130, 246, 0.3) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Result Cards ────────────────────────────────────────── */
.result-card-high {
    background: linear-gradient(135deg, rgba(190, 18, 60, 0.25), rgba(239, 68, 68, 0.15));
    border: 1px solid rgba(239, 68, 68, 0.5);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.3), 0 8px 32px rgba(0,0,0,0.4);
    backdrop-filter: blur(16px);
    animation: pulseRed 2s ease-in-out infinite;
}

.result-card-low {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.25), rgba(52, 211, 153, 0.15));
    border: 1px solid rgba(52, 211, 153, 0.5);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(52, 211, 153, 0.3), 0 8px 32px rgba(0,0,0,0.4);
    backdrop-filter: blur(16px);
    animation: pulseGreen 2s ease-in-out infinite;
}

@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.3), 0 8px 32px rgba(0,0,0,0.4); }
    50%       { box-shadow: 0 0 70px rgba(239, 68, 68, 0.55), 0 8px 32px rgba(0,0,0,0.4); }
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 40px rgba(52, 211, 153, 0.3), 0 8px 32px rgba(0,0,0,0.4); }
    50%       { box-shadow: 0 0 70px rgba(52, 211, 153, 0.55), 0 8px 32px rgba(0,0,0,0.4); }
}

.result-icon   { font-size: 4rem; margin-bottom: 0.5rem; }
.result-label  { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
.result-prob   { font-size: 1.1rem; opacity: 0.85; margin-bottom: 1rem; }
.result-advice { font-size: 0.9rem; opacity: 0.75; line-height: 1.6; }

/* ── Metric Cards (sidebar) ──────────────────────────────── */
.metric-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.metric-box {
    flex: 1;
    background: rgba(15, 25, 55, 0.6);
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.metric-val  { font-size: 1.3rem; font-weight: 700; color: var(--neon-cyan); }
.metric-lbl  { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }

/* ── Status Badge ────────────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-online  { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.badge-warning { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }

/* ── Health Tip Card ─────────────────────────────────────── */
.tip-card {
    background: rgba(15, 25, 55, 0.5);
    border-left: 3px solid var(--neon-cyan);
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
    color: var(--text-primary);
    line-height: 1.5;
}

/* ── Dividers ────────────────────────────────────────────── */
hr { border-color: var(--border-glass) !important; margin: 1rem 0 !important; }

/* ── Expander ────────────────────────────────────────────── */
details summary {
    color: var(--text-accent) !important;
    font-weight: 500 !important;
}

/* ── Checkbox ────────────────────────────────────────────── */
.stCheckbox > label {
    color: var(--text-accent) !important;
    font-weight: 500 !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.5); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler from disk, with fallback."""
    model_path  = "model/best_model.pkl"
    scaler_path = "model/scaler.pkl"

    model, scaler = None, None

    # Try model
    for path in [model_path, "outputs/best_model.pkl"]:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                break
            except Exception:
                pass

    # Try scaler
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    return model, scaler

model, scaler = load_model_and_scaler()


# ─────────────────────────────────────────────────────────────
# FEATURE ORDER  (must match training)
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "pregnancies", "glucose", "bloodpressure", "skinthickness",
    "insulin", "bmi", "diabetespedigreefunction", "age"
]


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def predict(inputs: dict):
    """Run prediction. Returns (label, probability, raw_pred)."""
    row = np.array([[inputs[f] for f in FEATURES]], dtype=float)
    if scaler is not None:
        row = scaler.transform(row)
    proba = model.predict_proba(row)[0][1]
    pred  = int(proba >= 0.5)
    return pred, proba


def bmi_from_hw(weight_kg: float, height_cm: float) -> float:
    """Compute BMI from weight (kg) and height (cm)."""
    h_m = height_cm / 100.0
    return round(weight_kg / (h_m ** 2), 2)


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + brand
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:2.5rem;'>🩺</div>
        <div style='font-family:"Orbitron",monospace; font-size:1.1rem; color:#63b3ed; font-weight:700; letter-spacing:2px;'>DiabetesScan</div>
        <div style='font-size:0.7rem; color:#718096; letter-spacing:3px; margin-top:2px;'>AI MEDICAL SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Model status
    if model is not None:
        st.markdown('<span class="status-badge badge-online">🟢 &nbsp;Model Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-warning">⚠️ &nbsp;Model Missing</span>', unsafe_allow_html=True)

    st.markdown("")

    # Navigation
    page = st.radio(
        "Navigation",
        ["🧬 Prediction Engine", "📊 About & Model", "💡 Health Tips"],
        label_visibility="collapsed"
    )

    st.divider()

    # Quick model stats
    st.markdown("""
    <div class='card-title'>📈 Model Performance</div>
    <div class='metric-row'>
        <div class='metric-box'>
            <div class='metric-val'>73%</div>
            <div class='metric-lbl'>Accuracy</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>81%</div>
            <div class='metric-lbl'>ROC-AUC</div>
        </div>
    </div>
    <div class='metric-row'>
        <div class='metric-box'>
            <div class='metric-val'>100</div>
            <div class='metric-lbl'>Estimators</div>
        </div>
        <div class='metric-box'>
            <div class='metric-val'>768</div>
            <div class='metric-lbl'>Trained on</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div style='font-size:0.7rem; color:#4a5568; text-align:center;'>⚕️ For informational purposes only.<br>Always consult a qualified physician.</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# PAGE 1 — PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
if page == "🧬 Prediction Engine":

    # Header
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>DiabetesScan AI</div>
        <div class='main-subtitle'>Advanced Diabetes Risk Assessment</div>
        <div class='header-line'></div>
    </div>
    """, unsafe_allow_html=True)

    # Model missing warning
    if model is None:
        st.warning(
            "⚠️ **Model file not found.** Please place `best_model.pkl` in the app directory. "
            "Predictions are unavailable until the model is loaded.",
            icon="🔍"
        )
        st.stop()

    # ── CARD 1: Metabolic Panel ──────────────────────────────
    st.markdown("""
    <div class='glass-card'>
        <div class='card-title'>🔬 Metabolic Panel</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        glucose = st.number_input(
            "💉 Glucose Level (mg/dL)",
            min_value=0.0, max_value=300.0, value=117.0, step=1.0,
            help="Plasma glucose concentration after 2-hour OGTT. Normal: 70–140 mg/dL"
        )

    with col2:
        blood_pressure = st.number_input(
            "❤️ Blood Pressure (mm Hg)",
            min_value=0.0, max_value=200.0, value=72.0, step=1.0,
            help="Diastolic blood pressure. Normal: 60–80 mm Hg"
        )

    with col3:
        insulin = st.number_input(
            "🧪 Insulin (µU/mL)",
            min_value=0.0, max_value=900.0, value=79.0, step=1.0,
            help="2-hour serum insulin. Normal fasting: <25 µU/mL"
        )

    # ── CARD 2: Body Metrics ─────────────────────────────────
    st.markdown("""
    <div class='glass-card'>
        <div class='card-title'>⚖️ Body Metrics</div>
    </div>
    """, unsafe_allow_html=True)

    bmi_unknown = st.checkbox("🤔 Don't know your BMI? Calculate it here", value=False)

    if bmi_unknown:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            weight_kg = st.number_input("⚖️ Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
        with c2:
            height_cm = st.number_input("📏 Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
        with c3:
            auto_bmi = bmi_from_hw(weight_kg, height_cm)
            st.markdown(f"""
            <div style='margin-top:1.8rem; background:rgba(99,179,237,0.12); border:1px solid rgba(99,179,237,0.3);
                        border-radius:10px; padding:0.7rem 1rem; text-align:center;'>
                <div style='font-size:0.7rem; color:#90cdf4; letter-spacing:2px;'>CALCULATED BMI</div>
                <div style='font-size:2rem; font-weight:700; color:#63b3ed;'>{auto_bmi}</div>
            </div>
            """, unsafe_allow_html=True)
        bmi = auto_bmi
    else:
        bmi = st.number_input(
            "🏋️ BMI (kg/m²)",
            min_value=0.0, max_value=80.0, value=32.0, step=0.1,
            help="Body Mass Index. Normal: 18.5–24.9"
        )

    col4, col5 = st.columns(2)
    with col4:
        skin_thickness = st.number_input(
            "📐 Skin Thickness (mm)",
            min_value=0.0, max_value=100.0, value=23.0, step=0.5,
            help="Triceps skin fold thickness. Normal: 12–35 mm"
        )
    with col5:
        st.markdown("") # spacing

    # ── CARD 3: Patient Profile ──────────────────────────────
    st.markdown("""
    <div class='glass-card'>
        <div class='card-title'>👤 Patient Profile</div>
    </div>
    """, unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3)

    with col6:
        pregnancies = st.number_input(
            "🤰 Pregnancies",
            min_value=0, max_value=20, value=3, step=1,
            help="Number of times pregnant"
        )

    with col7:
        age = st.number_input(
            "🎂 Age (years)",
            min_value=1, max_value=120, value=33, step=1,
            help="Patient age in years"
        )

    with col8:
        dpf = st.number_input(
            "🧬 Diabetes Pedigree Function",
            min_value=0.0, max_value=3.0, value=0.47, step=0.001, format="%.3f",
            help="Genetic diabetes risk score based on family history. Higher = more risk."
        )

    # ── PREDICT BUTTON ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("🔍 &nbsp; ANALYZE RISK", use_container_width=True)

    # ── RESULT DISPLAY ───────────────────────────────────────
    if predict_clicked:
        inputs = {
            "pregnancies":             float(pregnancies),
            "glucose":                 glucose,
            "bloodpressure":           blood_pressure,
            "skinthickness":           skin_thickness,
            "insulin":                 insulin,
            "bmi":                     bmi,
            "diabetespedigreefunction": dpf,
            "age":                     float(age),
        }

        with st.spinner("🔬 Running diagnostic analysis..."):
            import time; time.sleep(0.6)  # brief visual pause for UX
            pred, proba = predict(inputs)

        st.markdown("<br>", unsafe_allow_html=True)

        if pred == 1:
            pct = round(proba * 100, 1)
            st.markdown(f"""
            <div class='result-card-high'>
                <div class='result-icon'>❌</div>
                <div class='result-label' style='color:#fc8181;'>HIGH RISK</div>
                <div class='result-prob' style='color:#fbd38d;'>Diabetes Probability: <strong>{pct}%</strong></div>
                <hr style='border-color:rgba(239,68,68,0.3); margin:1rem auto; max-width:200px;'>
                <div class='result-advice'>
                    ⚠️ Our AI model indicates an <strong>elevated risk</strong> of diabetes.<br><br>
                    📋 <strong>Recommended Actions:</strong><br>
                    • Schedule an appointment with your physician immediately<br>
                    • Request a full HbA1c blood panel and fasting glucose test<br>
                    • Begin monitoring blood sugar levels daily<br>
                    • Review dietary habits and consider a nutrition consultation<br>
                    • Start or increase regular physical activity (150 min/week)
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pct = round((1 - proba) * 100, 1)
            st.markdown(f"""
            <div class='result-card-low'>
                <div class='result-icon'>✅</div>
                <div class='result-label' style='color:#68d391;'>LOW RISK</div>
                <div class='result-prob' style='color:#9ae6b4;'>Healthy Probability: <strong>{pct}%</strong></div>
                <hr style='border-color:rgba(52,211,153,0.3); margin:1rem auto; max-width:200px;'>
                <div class='result-advice'>
                    🎉 Our AI model indicates a <strong>low risk</strong> of diabetes at this time.<br><br>
                    💚 <strong>Keep it up — Maintain your health:</strong><br>
                    • Continue regular health check-ups (every 6–12 months)<br>
                    • Sustain a balanced diet rich in vegetables and whole grains<br>
                    • Stay active with at least 150 minutes of exercise weekly<br>
                    • Monitor your weight and BMI periodically<br>
                    • Stay hydrated and manage stress effectively
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Input summary expander
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 View Input Summary", expanded=False):
            summary = pd.DataFrame({
                "Feature":   ["Glucose", "Blood Pressure", "Insulin", "Skin Thickness",
                               "BMI", "Pregnancies", "Age", "Diabetes Pedigree Function"],
                "Your Value": [glucose, blood_pressure, insulin, skin_thickness,
                               bmi, pregnancies, age, dpf],
                "Normal Range": ["70–140 mg/dL", "60–80 mmHg", "<25 µU/mL", "12–35 mm",
                                 "18.5–24.9", "—", "—", "<0.5 (low risk)"],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2 — ABOUT & MODEL
# ─────────────────────────────────────────────────────────────
elif page == "📊 About & Model":

    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>About the System</div>
        <div class='main-subtitle'>Model Architecture & Dataset</div>
        <div class='header-line'></div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("""
        <div class='glass-card'>
            <div class='card-title'>🤖 Model Architecture</div>
            <p style='color:#a0aec0; font-size:0.9rem; line-height:1.7;'>
                This system uses a <strong style='color:#63b3ed;'>Random Forest Classifier</strong> — an ensemble
                of 100 decision trees trained on the Pima Indians Diabetes dataset. Each tree votes
                and the majority determines the final prediction.
            </p>
            <br>
            <table style='width:100%; font-size:0.85rem; border-collapse:collapse;'>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Algorithm</td>
                    <td style='color:#e2e8f0; font-weight:500;'>Random Forest (Ensemble)</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Estimators</td>
                    <td style='color:#e2e8f0; font-weight:500;'>100 Decision Trees</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Accuracy</td>
                    <td style='color:#68d391; font-weight:600;'>73.4%</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>ROC-AUC</td>
                    <td style='color:#68d391; font-weight:600;'>0.810</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Scaler</td>
                    <td style='color:#e2e8f0; font-weight:500;'>StandardScaler</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Balancing</td>
                    <td style='color:#e2e8f0; font-weight:500;'>Random Oversampling</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class='glass-card'>
            <div class='card-title'>📂 Dataset Info</div>
            <table style='width:100%; font-size:0.85rem; border-collapse:collapse;'>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Source</td>
                    <td style='color:#e2e8f0;'>Pima Indians Diabetes</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Samples</td>
                    <td style='color:#e2e8f0;'>768 patients</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Features</td>
                    <td style='color:#e2e8f0;'>8 clinical features</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Target</td>
                    <td style='color:#e2e8f0;'>Diabetic (1) / Not (0)</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Class Balance</td>
                    <td style='color:#f6ad55;'>65% neg / 35% pos</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Missing Data</td>
                    <td style='color:#e2e8f0;'>Median imputation</td>
                </tr>
                <tr>
                    <td style='color:#718096; padding:0.4rem 0;'>Outliers</td>
                    <td style='color:#e2e8f0;'>IQR winsorization</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance chart
    if model is not None:
        st.markdown("""
        <div class='glass-card'>
            <div class='card-title'>⚡ Feature Importance Rankings</div>
        </div>
        """, unsafe_allow_html=True)

        importance_df = pd.DataFrame({
            "Feature":    ["Glucose", "BMI", "Age", "Diabetes Pedigree Fn", "Pregnancies", "Insulin", "Blood Pressure", "Skin Thickness"],
            "Importance": sorted(model.feature_importances_, reverse=True)
        }).sort_values("Importance", ascending=True)

        import streamlit as st
        # Simple horizontal bar via st.bar_chart won't do horizontal; use st data
        st.bar_chart(
            importance_df.set_index("Feature")["Importance"],
            use_container_width=True,
            color="#63b3ed"
        )

    st.markdown("""
    <div class='glass-card'>
        <div class='card-title'>⚠️ Disclaimer</div>
        <p style='color:#a0aec0; font-size:0.88rem; line-height:1.7;'>
            This application is intended for <strong style='color:#f6ad55;'>educational and informational purposes only</strong>.
            It is <em>not</em> a substitute for professional medical advice, diagnosis, or treatment.
            Always seek the guidance of your physician or other qualified health provider with any
            questions you may have regarding a medical condition. Never disregard professional
            medical advice based on something you have read or seen in this app.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3 — HEALTH TIPS
# ─────────────────────────────────────────────────────────────
elif page == "💡 Health Tips":

    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>Health Tips</div>
        <div class='main-subtitle'>Diabetes Prevention & Management</div>
        <div class='header-line'></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🥗 Nutrition", "🏃 Exercise", "🧠 Lifestyle"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        tips_nutrition = [
            ("🥦", "Choose complex carbohydrates", "Opt for whole grains, legumes, and vegetables instead of refined carbs. They release glucose slowly, preventing sugar spikes."),
            ("🫐", "Load up on antioxidants", "Berries, leafy greens, and nuts contain compounds that improve insulin sensitivity and reduce inflammation."),
            ("💧", "Stay hydrated", "Drinking adequate water (2–3L daily) helps your kidneys flush excess glucose and keeps metabolism efficient."),
            ("🍽️", "Practice portion control", "Using smaller plates and mindful eating prevents overeating. Aim for balanced macros: 45–65% carbs, 20–35% fat, 10–35% protein."),
            ("🚫", "Limit added sugars", "Cut back on sugary drinks, processed snacks, and desserts. Check nutrition labels — sugar hides in many products."),
            ("🐟", "Include healthy fats", "Omega-3 fatty acids from salmon, flaxseed, and walnuts reduce inflammation and support cardiovascular health in diabetics."),
        ]
        for icon, title, desc in tips_nutrition:
            st.markdown(f"""
            <div class='tip-card'>
                <strong style='color:#76e4f7;'>{icon} {title}</strong><br>
                <span style='color:#a0aec0;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        tips_exercise = [
            ("🚶", "Daily walking", "A 30-minute brisk walk after meals significantly lowers blood sugar. Start with 10 minutes and build gradually."),
            ("🏋️", "Resistance training", "Strength training 2–3x/week builds muscle, which acts as a glucose sink — muscles absorb blood sugar without needing as much insulin."),
            ("🚴", "Aerobic exercise", "Cycling, swimming, or jogging for 150 minutes/week at moderate intensity is the gold standard for diabetes prevention."),
            ("🧘", "Yoga & flexibility", "Yoga reduces cortisol (stress hormone), which directly lowers blood glucose and improves insulin sensitivity."),
            ("⏱️", "Break sedentary time", "Stand or walk for 2–3 minutes every 30 minutes of sitting. Even micro-activity significantly improves glucose metabolism."),
        ]
        for icon, title, desc in tips_exercise:
            st.markdown(f"""
            <div class='tip-card'>
                <strong style='color:#68d391;'>{icon} {title}</strong><br>
                <span style='color:#a0aec0;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        tips_lifestyle = [
            ("😴", "Prioritize sleep", "Poor sleep raises cortisol and ghrelin, leading to insulin resistance. Aim for 7–9 hours of quality sleep per night."),
            ("🧘", "Manage stress", "Chronic stress elevates blood glucose directly. Try meditation, deep breathing, or 10-minute mindfulness sessions daily."),
            ("🚭", "Avoid smoking", "Smoking increases insulin resistance by 30–40%. Quitting dramatically reduces your risk of diabetic complications."),
            ("📊", "Monitor regularly", "Regular blood glucose monitoring lets you understand how food and activity affect your levels. Knowledge is power."),
            ("👨‍⚕️", "Annual check-ups", "See your doctor for HbA1c, lipid panels, kidney function, and eye exams — essential for early detection and management."),
            ("🤝", "Build a support system", "Social support improves adherence to healthy habits. Consider joining a diabetes prevention program or support group."),
        ]
        for icon, title, desc in tips_lifestyle:
            st.markdown(f"""
            <div class='tip-card'>
                <strong style='color:#f6ad55;'>{icon} {title}</strong><br>
                <span style='color:#a0aec0;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)
