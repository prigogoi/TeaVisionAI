import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import pandas as pd
from datetime import datetime
import time
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TeaVision AI",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "camera_file" not in st.session_state:
    st.session_state.camera_file = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: white;
}

.stApp {
    background:
        radial-gradient(circle at top left, #1b4332 0%, transparent 30%),
        radial-gradient(circle at bottom right, #081c15 0%, transparent 30%),
        linear-gradient(
            135deg,
            #020b07 0%,
            #071b14 45%,
            #0b2f20 100%
        );

    color: white;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 1rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

.main-title {
    text-align: center;
    font-size: 75px;
    font-weight: 800;
    margin-top: 10px;

    background:
        linear-gradient(
            90deg,
            #dcfce7,
            #4ade80,
            #16a34a
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #d1fae5;
    font-size: 22px;
    margin-top: -10px;
    margin-bottom: 40px;
}

div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-bottom: 50px;
}

div[role="radiogroup"] > label {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.03)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 22px;
    padding: 15px 28px;
    min-width: 220px;
    text-align: center;
    backdrop-filter: blur(14px);
    transition: 0.35s ease;

    box-shadow:
        0 10px 25px rgba(0,0,0,0.25);
}

div[role="radiogroup"] > label:hover {
    transform: translateY(-5px);

    border:
        1px solid rgba(74,222,128,0.4);

    box-shadow:
        0 15px 35px rgba(34,197,94,0.25);
}

div[role="radiogroup"] p {
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

.card {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.03)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 28px;
    padding: 35px;
    text-align: center;
    backdrop-filter: blur(16px);
    transition: 0.3s;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.3);

    height: 100%;
}

.card:hover {
    transform: translateY(-8px);

    box-shadow:
        0 20px 40px rgba(34,197,94,0.18);
}

.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 18px;
    border: none;

    background:
        linear-gradient(
            90deg,
            #22c55e,
            #4ade80
        );

    color: #04130c;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s ease;

    box-shadow:
        0 8px 25px rgba(34,197,94,0.35);
}

.stButton > button:hover {
    transform: translateY(-3px);

    box-shadow:
        0 12px 30px rgba(34,197,94,0.45);
}

[data-testid="stFileUploader"] {
    background:
        rgba(255,255,255,0.05);

    border:
        1px dashed rgba(255,255,255,0.15);

    border-radius: 22px;
    padding: 25px;
    backdrop-filter: blur(10px);
}

img {
    border-radius: 24px !important;

    box-shadow:
        0 15px 40px rgba(0,0,0,0.4);
}

[data-testid="stAlert"] {
    border-radius: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.fc.in_features,
            7
        )
    )

    model.load_state_dict(
        torch.load(
            "ResNet50.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

disease_model = load_model()

# =========================================================
# LABELS
# =========================================================

disease_class_names = [

    "Tea Algal Leaf Spot",
    "Brown Blight",
    "Gray Blight",
    "Helopeltis",
    "Red Spider",
    "Green Mirid Bug",
    "Healthy Leaf"
]

# =========================================================
# DISEASE DATA
# =========================================================

disease_data = {

    "Tea Algal Leaf Spot": {
        "description": "Algal infection producing reddish-orange spots.",
        "treatment": "Apply copper fungicide and improve airflow.",
        "prevention": "Avoid excessive moisture."
    },

    "Brown Blight": {
        "description": "Fungal disease causing brown patches.",
        "treatment": "Use recommended fungicides.",
        "prevention": "Remove infected leaves."
    },

    "Gray Blight": {
        "description": "Gray fungal lesions on leaves.",
        "treatment": "Spray fungicide regularly.",
        "prevention": "Maintain field hygiene."
    },

    "Helopeltis": {
        "description": "Insect attack causing dark feeding marks.",
        "treatment": "Use insecticides carefully.",
        "prevention": "Monitor tea bushes regularly."
    },

    "Red Spider": {
        "description": "Mite infestation causing drying.",
        "treatment": "Apply miticides.",
        "prevention": "Maintain humidity levels."
    },

    "Green Mirid Bug": {
        "description": "Pest attack damaging leaves.",
        "treatment": "Use approved pesticides.",
        "prevention": "Regular inspection."
    },

    "Healthy Leaf": {
        "description": "Leaf appears healthy.",
        "treatment": "No treatment required.",
        "prevention": "Continue proper care."
    }
}

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(img_path):

    transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(img_path).convert("RGB")

    return transform(image).unsqueeze(0)

# =========================================================
# PREDICTION
# =========================================================

def predict_disease(img_path):

    input_tensor = preprocess_image(img_path)

    with torch.no_grad():

        output = disease_model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

        predicted_class = disease_class_names[
            predicted_idx.item()
        ]

        confidence_score = confidence.item() * 100

        return predicted_class, confidence_score

# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div class="main-title">
        TeaVision AI
    </div>

    <div class="subtitle">
        Deep Learning Powered Tea Leaf Disease Detection
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(

    "",

    [
        "🌿 Home",
        "✨ Detect Disease",
        "📜 Prediction History",
        "🔮 About"
    ],

    horizontal=True
)

# =========================================================
# HOME
# =========================================================

if page == "🌿 Home":

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="card">
            <h1>⚡</h1>
            <h2>Fast Detection</h2>
            <p>Instant AI powered disease analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">
            <h1>📷</h1>
            <h2>Camera Support</h2>
            <p>Capture tea leaf images directly.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="card">
            <h1>🧠</h1>
            <h2>Deep Learning</h2>
            <p>Powered by ResNet50 architecture.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# DETECT PAGE
# =========================================================

elif page == "✨ Detect Disease":

    col1, col2 = st.columns(2)

    with col1:

        uploaded_image = st.file_uploader(
            "Upload Tea Leaf Image",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.uploader_key}"
        )

    with col2:

        camera_image = st.camera_input(
            "Capture Tea Leaf"
        )

    if uploaded_image is not None:

        st.session_state.uploaded_file = uploaded_image
        st.session_state.camera_file = None

    if camera_image is not None:

        st.session_state.camera_file = camera_image
        st.session_state.uploaded_file = None

    image = None

    if st.session_state.uploaded_file is not None:

        image = Image.open(
            st.session_state.uploaded_file
        ).convert("RGB")

        save_path = os.path.join(
            "uploaded",
            st.session_state.uploaded_file.name
        )

    elif st.session_state.camera_file is not None:

        image = Image.open(
            st.session_state.camera_file
        ).convert("RGB")

        save_path = os.path.join(
            "uploaded",
            "camera_capture.jpg"
        )

    if image is not None:

        os.makedirs("uploaded", exist_ok=True)

        image.save(save_path)

        left, center, right = st.columns([1,2,1])

        with center:

            st.image(
                image,
                caption="Tea Leaf Preview",
                use_container_width=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            b1, b2 = st.columns(2)

            with b1:

                detect = st.button(
                    "✨ Analyze Leaf"
                )

            with b2:

                clear = st.button(
                    "🗑 Clear Image"
                )

                if clear:

                    st.session_state.uploaded_file = None
                    st.session_state.camera_file = None

                    st.session_state.uploader_key += 1

                    if os.path.exists(save_path):
                        os.remove(save_path)

                    st.rerun()

            if detect:

                loading = st.empty()

                loading.markdown("""
                <div class="card">
                    <h2>✨ AI Vision Scanning Tea Leaf...</h2>
                    <p>Analyzing disease patterns using Deep Learning.</p>
                </div>
                """, unsafe_allow_html=True)

                progress = st.progress(0)

                for i in range(100):

                    time.sleep(0.01)

                    progress.progress(i + 1)

                predicted_class, confidence = predict_disease(save_path)

                loading.empty()
                progress.empty()

                if confidence > 85:
                    severity = "Severe"

                elif confidence > 60:
                    severity = "Moderate"

                else:
                    severity = "Mild"

                st.session_state.history.append({

                    "Disease":
                    predicted_class,

                    "Confidence":
                    round(confidence, 2),

                    "Severity":
                    severity,

                    "Time":
                    datetime.now().strftime(
                        "%d %b %Y | %I:%M %p"
                    )
                })

                st.success(
                    f"Prediction: {predicted_class}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.metric(
                    "Severity",
                    severity
                )

                st.info(
                    disease_data[predicted_class]["description"]
                )

                st.success(
                    f"Treatment: {disease_data[predicted_class]['treatment']}"
                )

                st.warning(
                    f"Prevention: {disease_data[predicted_class]['prevention']}"
                )

# =========================================================
# HISTORY
# =========================================================

elif page == "📜 Prediction History":

    if len(st.session_state.history) == 0:

        st.warning(
            "No predictions available yet."
        )

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True
        )

# =========================================================
# ABOUT
# =========================================================

elif page == "🔮 About":

    st.markdown("""
    <div class="card">

    <h1>🌿 TeaVision AI</h1>

    <p style="font-size:18px; line-height:1.8;">

    TeaVision AI is an advanced Deep Learning
    powered agricultural assistant built to
    detect tea leaf diseases using Artificial
    Intelligence and Computer Vision.

    </p>

    <h2>✨ Features</h2>

    <p style="line-height:2; font-size:17px;">

    ✅ AI Disease Detection <br>
    ✅ Severity Analysis <br>
    ✅ Treatment Guidance <br>
    ✅ Prevention Suggestions <br>
    ✅ Prediction History <br>
    ✅ Live Camera Support <br>
    ✅ Futuristic Premium UI

    </p>

    <h2>👨‍💻 Developer</h2>

    <p style="font-size:18px;">
    Priyam Gogoi
    </p>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""

<br><br>

<hr style="
    border:1px solid rgba(255,255,255,0.08);
">

<center>

<p style="
    color:#b7d8c2;
    font-size:15px;
">

🌿 TeaVision AI © 2026

<br>

Deep Learning Powered Agricultural Assistant

</p>

</center>

""", unsafe_allow_html=True)