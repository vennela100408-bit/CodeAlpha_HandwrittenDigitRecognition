import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DigitVision AI",
    page_icon="✍️",
    layout="wide"
)

MODEL_PATH = Path("models/mnist_cnn.keras")


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(124, 58, 237, 0.18), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(59, 130, 246, 0.15), transparent 25%),
        linear-gradient(135deg, #f8f7ff 0%, #eef2ff 50%, #f5f3ff 100%);
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Hero */
.hero {
    background: linear-gradient(
        135deg,
        #4c1d95,
        #6d28d9,
        #7c3aed
    );

    padding: 42px 45px;
    border-radius: 28px;
    color: white;
    box-shadow: 0 20px 45px rgba(76, 29, 149, 0.25);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 17px;
    opacity: 0.9;
    margin: 0;
}

/* Section title */
.section-title {
    color: #312e81;
    font-size: 24px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Upload box */
.upload-card {
    background: rgba(255,255,255,0.88);
    border: 2px dashed #8b5cf6;
    border-radius: 22px;
    padding: 28px;
    box-shadow: 0 10px 30px rgba(76,29,149,0.08);
}

/* Result card */
.result-card {
    background: white;
    border-radius: 24px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(49,46,129,0.12);
    border: 1px solid #e9d5ff;
}

.result-label {
    color: #6b7280;
    font-size: 15px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-digit {
    font-size: 86px;
    font-weight: 800;
    color: #6d28d9;
    line-height: 1.1;
    margin: 8px 0;
}

/* Confidence metric numbers */
[data-testid="stMetricValue"] {
    color: #312e81 !important;
    font-weight: 800 !important;
}

/* Confidence metric labels */
[data-testid="stMetricLabel"] {
    color: #4c1d95 !important;
    font-weight: 600 !important;
}

/* Info cards */
.info-card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #ddd6fe;
    box-shadow: 0 8px 25px rgba(76,29,149,0.07);
}

.info-card h4 {
    color: #4c1d95;
    margin-bottom: 8px;
}

.info-card p {
    color: #6b7280;
    margin: 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<h1>✍️ DigitVision AI</h1>

<p>
Smart Handwritten Digit Recognition powered by
Convolutional Neural Networks
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INTRO
# =========================================================

st.markdown("""
<div class="info-card">

<h4>🧠 How does it work?</h4>

<p>
Upload an image containing a handwritten digit.
The trained CNN model processes the image and identifies
which digit from 0 to 9 it most likely represents.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")


# =========================================================
# MODEL CHECK
# =========================================================

if not MODEL_PATH.exists():

    st.error(
        "⚠️ AI model is not available yet."
    )

    st.info(
        "Run `python train_model.py` first. "
        "It will automatically create the `models` folder "
        "and the `mnist_cnn.keras` model."
    )

    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload Your Digit</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a handwritten digit image",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image containing one handwritten digit."
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("L")

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-title">🖼️ Your Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            width=300
        )


    # -----------------------------------------------------
    # PROCESS
    # -----------------------------------------------------

    image_array = np.array(image)

    # Convert bright-background image to MNIST-style
    # dark digit on light background
    if image_array.mean() > 127:

        image_array = 255 - image_array


    # Resize
    image_28 = Image.fromarray(
        image_array.astype(np.uint8)
    ).resize(
        (28, 28)
    )


    # Normalize
    processed = (
        np.array(image_28)
        .astype("float32")
        / 255.0
    )


    # CNN input shape
    processed = processed.reshape(
        1,
        28,
        28,
        1
    )


    # Prediction
    probabilities = model.predict(
        processed,
        verbose=0
    )[0]


    predicted_digit = int(
        np.argmax(probabilities)
    )


    confidence = (
        float(probabilities[predicted_digit])
        * 100
    )


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-title">🤖 AI Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Predicted Digit
                </div>

                <div class="result-digit">
                    {predicted_digit}
                </div>

                <div class="confidence">
                    Confidence: {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.write("")

    st.markdown(
        '<div class="section-title">📊 AI Confidence by Digit</div>',
        unsafe_allow_html=True
    )

    probability_columns = st.columns(5)

    for digit in range(10):

        probability = float(
            probabilities[digit]
        )

        column = probability_columns[
            digit % 5
        ]

        with column:

            st.metric(
                label=f"Digit {digit}",
                value=f"{probability * 100:.1f}%"
            )

            st.progress(
                probability
            )


else:

    st.write("")

    st.markdown("""
    <div class="info-card">

    <h4>👆 Ready for prediction</h4>

    <p>
    Upload a handwritten digit image above.
    The AI will analyze it and display its prediction
    together with the confidence score.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.write("")
st.write("")

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown("""
    <div class="info-card">

    <h4>🧠 CNN Model</h4>

    <p>
    Convolutional Neural Network trained
    for handwritten digit classification.
    </p>

    </div>
    """, unsafe_allow_html=True)


with info2:

    st.markdown("""
    <div class="info-card">

    <h4>🔢 MNIST Dataset</h4>

    <p>
    Thousands of handwritten digit images
    representing numbers from 0 to 9.
    </p>

    </div>
    """, unsafe_allow_html=True)


with info3:

    st.markdown("""
    <div class="info-card">

    <h4>⚡ AI Prediction</h4>

    <p>
    The model analyzes the uploaded image
    and selects the most probable digit.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

CodeAlpha Machine Learning Internship • Task 3

</div>
""", unsafe_allow_html=True)