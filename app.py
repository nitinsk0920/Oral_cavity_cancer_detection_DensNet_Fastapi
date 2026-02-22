import streamlit as st
import requests

st.title("🩺 Oral Cancer Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={"file": uploaded_file.getvalue()}
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
                st.info(f"Confidence: {result['confidence']*100:.2f}%")
            else:
                st.error("Error occurred")