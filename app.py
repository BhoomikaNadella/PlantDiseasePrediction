import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from src.precautions import precautions_dict
import tempfile
import os

# 🌿 Load model
MODEL_PATH = "models/model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# 🌾 Define class labels (same as your training folders)
CLASS_NAMES = [
    "Apple___Black_rot", "Apple___healthy", "Corn___Common_rust",
    "Corn___healthy", "Grape___Black_rot", "Potato___Early_blight",
    "Potato___healthy", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___healthy", "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Yellow_Leaf_Curl_Virus"
]

# 🌸 Prediction function
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    return predicted_class, confidence

# 🌿 Streamlit UI
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

st.title("🌿 Plant Disease Detection & Precaution Advisor")
st.write("Upload an image or take a photo to identify plant diseases and get care tips!")

# Tabs for file upload and webcam
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Use Webcam"])

# ---- Upload tab ----
with tab1:
    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        st.image(temp_file.name, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict from Uploaded Image"):
            with st.spinner("Analyzing... please wait ⏳"):
                label, confidence = predict_disease(temp_file.name)
                st.success(f"🌿 Predicted: **{label}**")
                st.info(f"💯 Confidence: **{confidence * 100:.2f}%**")

                # Show precaution
                precaution = precautions_dict.get(label, "No data available for this disease.")
                st.warning(f"💡 Precaution: {precaution}")

# ---- Webcam tab ----
with tab2:
    camera_input = st.camera_input("Take a photo using your webcam")

    if camera_input is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(camera_input.getbuffer())
        st.image(temp_file.name, caption="Captured Image", use_container_width=True)

        if st.button("🔍 Predict from Camera Image"):
            with st.spinner("Analyzing your plant... 🌿"):
                label, confidence = predict_disease(temp_file.name)
                st.success(f"🌿 Predicted: **{label}**")
                st.info(f"💯 Confidence: **{confidence * 100:.2f}%**")

                precaution = precautions_dict.get(label, "No data available for this disease.")
                st.warning(f"💡 Precaution: {precaution}")
