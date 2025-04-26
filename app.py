import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('brain_tumor_detector.h5')

st.title("🧠 Brain Tumor Detection from MRI")

uploaded_file = st.file_uploader("Upload an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded MRI.', use_column_width=True)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("Result: Tumor Detected 🧠⚠️")
    else:
        st.success("Result: No Tumor Detected ✅")