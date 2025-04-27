import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from fpdf import FPDF
import os
from datetime import datetime

# ========== Page Configuration ==========
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# ========== Sidebar ==========
with st.sidebar:
    st.title("🧠 Brain Tumor Detection")
    st.markdown("Upload MRI images to detect brain tumors using AI.")
    st.markdown("---")
    st.write("Developed with ❤️ using Streamlit, TensorFlow, and FPDF.")

# ========== Load Model ==========
@st.cache_resource
def load_prediction_model():
    model = load_model('your_model.h5')  # Replace 'your_model.h5' with your model filename
    return model

model = load_prediction_model()

# ========== Helper Functions ==========
def predict_tumor(img):
    img = img.resize((150, 150))  # Update if your model expects different size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return prediction

def generate_pdf(patient_name, prediction_result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Brain Tumor Detection Report", ln=True, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Prediction Result: {prediction_result}", ln=True)
    
    output_path = "report.pdf"
    pdf.output(output_path)
    return output_path

# ========== Main Layout ==========
st.title("🔍 Brain Tumor Detection from MRI Scans")

# Step 1: Input Patient Details
patient_name = st.text_input("Enter Patient's Name", "")

# Step 2: Upload Image
uploaded_file = st.file_uploader("Upload an MRI Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file and patient_name:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Scan", use_column_width=True)
    
    if st.button("🧠 Predict Tumor"):
        with st.spinner("Analyzing the MRI Scan..."):
            prediction = predict_tumor(img)
            if prediction[0][0] > 0.5:
                result = "No Tumor Detected 😊"
                st.success(result)
            else:
                result = "Tumor Detected ⚠️"
                st.error(result)
        
        # Step 3: Generate PDF Report
        pdf_file_path = generate_pdf(patient_name, result)
        with open(pdf_file_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Report",
                data=pdf_file,
                file_name=f"{patient_name}_brain_tumor_report.pdf",
                mime="application/pdf"
            )

elif uploaded_file and not patient_name:
    st.warning("⚠️ Please enter the patient's name before prediction!")

# ========== Footer ==========
st.markdown("---")
st.caption("© 2025 Brain Tumor Detection Project | Made with Streamlit 🚀")