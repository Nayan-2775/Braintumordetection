import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from fpdf import FPDF
from datetime import datetime

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ========== Sidebar ==========
with st.sidebar:
    st.title("🧠 Brain Tumor Detection")
    st.markdown("Upload MRI images to detect brain tumors using AI.")
    st.markdown("---")
    st.write("Developed with ❤️ using Streamlit, TensorFlow, and FPDF.")

# ========== Load Model ==========
@st.cache_resource
def load_prediction_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_detector.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_prediction_model()
if model is None:
    st.stop()

# ========== Helper Functions ==========
def predict_tumor(uploaded_file):
    try:
        # Use Keras preprocessing to match training pipeline
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array)
        return prediction, img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def generate_pdf(patient_name, prediction_result):
    try:
        # Replace emojis to avoid encoding issues
        prediction_result = prediction_result.replace("🧠⚠️", "(Tumor Detected)").replace("✅", "(No Tumor)")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Brain Tumor Detection Report", ln=True, align="C")
        
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Prediction Result: {prediction_result}", ln=True)
        
        output_path = f"{patient_name}_brain_tumor_report.pdf"
        pdf.output(output_path)
        return output_path
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# ========== Main Layout ==========
st.title("🔍 Brain Tumor Detection from MRI Scans")

# Step 1: Input Patient Details
patient_name = st.text_input("Enter Patient's Name", "")

# Step 2: Upload Image
uploaded_file = st.file_uploader("Upload an MRI Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file and patient_name:
    prediction, img = predict_tumor(uploaded_file)
    
    if prediction is not None and img is not None:
        st.image(img, caption="Uploaded MRI Scan", use_container_width=True)
        
        if st.button("🧠 Predict Tumor"):
            with st.spinner("Analyzing the MRI Scan..."):
                # Align prediction logic with old code
                if prediction[0][0] > 0.5:
                    result = "Result: Tumor Detected 🧠⚠️"
                    st.error(result)
                else:
                    result = "Result: No Tumor Detected ✅"
                    st.success(result)
                
                # Generate PDF Report
                pdf_file_path = generate_pdf(patient_name, result)
                if pdf_file_path:
                    with open(pdf_file_path, "rb") as pdf_file:
                        st.download_button(
                            label="📄 Download Report",
                            data=pdf_file,
                            file_name=f"{patient_name}_brain_tumor_report.pdf",
                            mime="application/pdf"
                        )

elif uploaded_file and not patient_name:
    st.warning("⚠️ Please enter the patient's name before prediction!")
elif not uploaded_file and patient_name:
    st.warning("⚠️ Please upload an MRI scan image!")

# ========== Footer ==========
st.markdown("---")
st.caption("© 2025 Brain Tumor Detection Project | Made with Streamlit 🚀")