import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.markdown("""
<style>
/* General Body and Main Container Styling for Dark Theme */
body {
    background-color: #1a1a1a; /* Dark background */
    color: #e0e0e0; /* Light text */
    font-family: 'Roboto', sans-serif; /* Elegant font, fallback to sans-serif */
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 700px; /* Limit width for centered content */
    margin: auto; /* Center content */
}

/* Header Styling */
h1, h2, h3, h4, h5, h6 {
    color: #4CAF50; /* Eye-catching green for headers */
    font-weight: bold;
    text-align: center; /* Center headers */
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #333; /* Subtle line below headers */
    margin-bottom: 1.5rem;
}

/* Buttons */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease-in-out;
    width: 100%; /* Make buttons span full width */
    margin-top: 1rem;
}

.stButton>button:hover {
    background-color: #5cb85c;
    box-shadow: 0 6px 12px rgba(0, 255, 0, 0.4); /* Green glow effect */
    transform: translateY(-2px);
}

/* File Uploader and Camera Input Preview Styling */
.stFileUploader, .stCameraInput {
    border: 2px dashed #4CAF50;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    background-color: #2a2a2a;
    transition: all 0.3s ease;
    margin-bottom: 1.5rem;
}

.stFileUploader:hover, .stCameraInput:hover {
    background-color: #3a3a3a;
    border-color: #66bb6a;
}

.stImage { /* Style for displayed images */
    border-radius: 12px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
    margin: 1.5rem auto; /* Center image and provide space */
    display: block; /* Ensure margin auto works */
    max-width: 100%;
    height: auto;
}

/* Text and general elements */
p, label, .stMarkdown, .css-1jc7o2r, .css-1l0bqyk, .css-1vq4p4u, .css-1qxtsq5, .css-1d391kg {
    color: #c0c0c0;
    line-height: 1.6;
}

/* Success/Error messages */
.stAlert {
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
.stAlert.success {
    background-color: #d4edda;
    color: #155724;
}
.stAlert.error {
    background-color: #f8d7da;
    color: #721c24;
}

/* Streamlit's main content area */
.stApp {
    background-color: #1a1a1a;
}

/* Responsiveness */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Load the pre-trained model
# Instead of deserializing the full model (which can be brittle across TF/Keras versions),
# we rebuild the architecture in code (as in the training notebook) and load the trained weights.
WEIGHTS_PATH = 'fruit_fresh_spoiled_weights.h5'

@st.cache_resource
def load_my_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None,  # weights will be restored from WEIGHTS_PATH
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.load_weights(WEIGHTS_PATH)
    return model

model = load_my_model()

IMG_SIZE = (150, 150)

def classify_fruit_streamlit(img):
    """Predicts whether a fruit image is fresh or spoiled."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(x)[0][0]

    if prediction < 0.5:
        label = "🍏 Fresh"
        confidence = 1 - prediction
    else:
        label = "🍂 Spoiled"
        confidence = prediction

    return label, confidence

# Streamlit App Layout
st.title("🍎 Fresh vs Spoiled Fruit Detector")
st.write("Upload an image of a fruit or use your camera to classify it as fresh or spoiled.")

# Input selection radio button
input_source = st.radio(
    "Select input source:",
    ('Upload Picture', 'Camera Input'),
    key="input_selector"
)

uploaded_file = None
camera_image = None

if input_source == 'Upload Picture':
    uploaded_file = st.file_uploader("Choose an image from your files...", type=["jpg", "jpeg", "png"])
elif input_source == 'Camera Input':
    camera_image = st.camera_input("Take a picture with your camera")

image_to_process = None
if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file)
elif camera_image is not None:
    image_to_process = Image.open(camera_image)

if image_to_process is not None:
    st.image(image_to_process, caption='Selected Image', use_container_width=True)
    st.write("")

    label, confidence = classify_fruit_streamlit(image_to_process)

    st.write("### Prediction:")
    if "Fresh" in label:
        st.success(f"The fruit is {label} with {confidence*100:.2f}% confidence.")
    else:
        st.error(f"The fruit is {label} with {confidence*100:.2f}% confidence.")
