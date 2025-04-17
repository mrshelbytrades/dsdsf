import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gdown
import os

@st.cache_resource
def download_and_load_model():
    model_file = "tumor_model.keras"
    if not os.path.exists(model_file):
        gdown.download(
            url="https://drive.google.com/uc?id=14eEJk1IZCrtraKp5I3QnhajwbOcb-z8y",
            output=model_file,
            quiet=False
        )
    return load_model(model_file)

model = download_and_load_model()

st.title("🧠 Brain Tumor Detection App")
st.markdown("<h2>Upload an MRI scan to check for brain tumor using a deep learning model by @ Mohsin💻</h2>", unsafe_allow_html=True)

st.markdown("<h3>Upload an image (JPG/PNG) 🖼️</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    image = image.resize((224, 224))
    img = img_to_array(image)
   
    img_array = np.expand_dims(img, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)

    # Output the prediction
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability

    # Map the prediction to a label
    if predicted_class == 0:
        result = "No Tumor Detected ✅"
    else:
        result = "Tumor Detected ⚠️"

    st.subheader("Prediction Result:")
    st.success(f"🎯 {result}")

    tumor_prob = prediction[0][1] * 100  # Probability of Tumor
    no_tumor_prob = prediction[0][0] * 100  # Probability of No Tumor

    if tumor_prob > no_tumor_prob:
        st.markdown(f"""
        <h3 style="font-size: 24px; font-weight: bold;">Probability:</h3>
        <p style="font-size: 22px; color: red;">{tumor_prob:.2f}% 🔴📊</p>

        <h3 style="font-size: 24px; font-weight: bold;">Interpretation:</h3>
        <p style="font-size: 20px;">🧐🧠 The model predicts with **{tumor_prob:.2f}%** confidence that the image contains a tumor,  
        based on features associated with tumor patterns.</p>

        <h3 style="font-size: 24px; font-weight: bold;">Recommendation:</h3>
        <p style="font-size: 20px;">🚨 While the model shows a high probability of a tumor, don't panic!🫂  
        You should consult a healthcare professional 🩺 for further tests and treatment.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <h3 style="font-size: 24px; font-weight: bold;">Probability:</h3>
        <p style="font-size: 22px; color: green;">{no_tumor_prob:.2f}% 🟢📊</p>

        <h3 style="font-size: 24px; font-weight: bold;">Interpretation:</h3>
        <p style="font-size: 20px;">😌💚 The model predicts with **{no_tumor_prob:.2f}%** confidence that the image does not contain a tumor.  
        It has identified characteristics typical of non-tumor regions.</p>

        <h3 style="font-size: 24px; font-weight: bold;">Recommendation:</h3>
        <p style="font-size: 20px;">😎 Chill, you are not going to die! 💪 The model has your back.  
        But hey, if you still feel like stressing over it, maybe just grab a snack and relax 🍿</p>
        """, unsafe_allow_html=True)
